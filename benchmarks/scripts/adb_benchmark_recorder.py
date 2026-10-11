#!/usr/bin/env python3
"""
OfflineDocAI - Real-Time USB / ADB Benchmark Harness
Captures on-device RAG telemetry across 4GB, 6GB, and 8GB RAM devices.
"""

import os
import sys
import time
import json
import re
import csv
import subprocess
import threading
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PACKAGE_NAME = "com.siddu.offlinedocai"
BENCHMARK_TAG = "OFFLINE_DOC_AI_BENCHMARK"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

class BenchmarkRecorder:
    def __init__(self, serial=None):
        self.serial = serial
        self.device_info = {}
        self.ram_tier = "tier_unknown"
        self.stop_signal = threading.Event()
        self.latest_hardware_snapshot = {
            "pss_total_mb": 0.0,
            "native_heap_mb": 0.0,
            "java_heap_mb": 0.0,
            "cpu_util_pct": 0.0,
            "battery_temp_c": 0.0
        }

    def adb_cmd(self, cmd_args):
        base = ["adb"]
        if self.serial:
            base.extend(["-s", self.serial])
        return subprocess.run(base + cmd_args, capture_output=True, text=True, errors="replace")

    def detect_device(self):
        print("[*] Detecting connected Android devices via ADB...")
        res = self.adb_cmd(["devices"])
        lines = [line.strip() for line in res.stdout.strip().split("\n")[1:] if line.strip() and not line.startswith("*")]
        if not lines:
            print("[!] ERROR: No Android device found via USB ADB.")
            print("    Please ensure USB debugging is enabled on your phone and connect via USB.")
            return False

        if not self.serial:
            self.serial = lines[0].split()[0]

        print(f"[+] Connected to device serial: {self.serial}")

        # Fetch Properties
        model = self.adb_cmd(["shell", "getprop", "ro.product.model"]).stdout.strip()
        brand = self.adb_cmd(["shell", "getprop", "ro.product.brand"]).stdout.strip()
        os_ver = self.adb_cmd(["shell", "getprop", "ro.build.version.release"]).stdout.strip()
        sdk_ver = self.adb_cmd(["shell", "getprop", "ro.build.version.sdk"]).stdout.strip()
        soc = self.adb_cmd(["shell", "getprop", "ro.board.platform"]).stdout.strip()

        # Fetch Total RAM
        meminfo_out = self.adb_cmd(["shell", "cat", "/proc/meminfo"]).stdout
        total_ram_kb = 0
        for mline in meminfo_out.split("\n"):
            if "MemTotal" in mline:
                parts = mline.split()
                if len(parts) >= 2:
                    total_ram_kb = int(parts[1])
                break
        total_ram_mb = total_ram_kb / 1024.0
        total_ram_gb = total_ram_mb / 1024.0

        if total_ram_gb <= 4.8:
            self.ram_tier = "tier_4gb"
        elif total_ram_gb <= 6.8:
            self.ram_tier = "tier_6gb"
        else:
            self.ram_tier = "tier_8gb"

        self.device_info = {
            "serial": self.serial,
            "brand": brand,
            "model": model,
            "android_version": f"Android {os_ver} (API {sdk_ver})",
            "soc_chipset": soc,
            "total_ram_mb": round(total_ram_mb, 1),
            "total_ram_gb": round(total_ram_gb, 2),
            "ram_tier": self.ram_tier
        }

        print("\n" + "="*60)
        print(" DEVICE PROFILE DETECTED")
        print("="*60)
        for k, v in self.device_info.items():
            print(f"  • {k:20s}: {v}")
        print("="*60 + "\n")
        return True

    def poll_hardware_stats(self):
        """Background poller for memory, CPU, and temperature."""
        while not self.stop_signal.is_set():
            try:
                # 1. dumpsys meminfo
                mem_res = self.adb_cmd(["shell", "dumpsys", "meminfo", PACKAGE_NAME])
                pss_total = 0.0
                native_heap = 0.0
                java_heap = 0.0

                for line in mem_res.stdout.split("\n"):
                    line_clean = line.strip()
                    if "TOTAL PSS:" in line:
                        m = re.search(r"TOTAL PSS:\s+(\d+)", line)
                        if m:
                            pss_total = int(m.group(1)) / 1024.0
                    elif line_clean.startswith("Native Heap") or "Native Heap" in line:
                        parts = line.split()
                        for p in parts:
                            if p.isdigit() and int(p) > 100:
                                native_heap = int(p) / 1024.0
                                break
                    elif line_clean.startswith("Java Heap") or "Java Heap" in line:
                        parts = line.split()
                        for p in parts:
                            if p.isdigit() and int(p) > 100:
                                java_heap = int(p) / 1024.0
                                break

                # 2. Battery Temperature
                batt_res = self.adb_cmd(["shell", "dumpsys", "battery"])
                temp_c = 0.0
                for line in batt_res.stdout.split("\n"):
                    if "temperature:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2 and parts[1].strip().isdigit():
                            temp_c = int(parts[1].strip()) / 10.0
                            break

                self.latest_hardware_snapshot = {
                    "pss_total_mb": round(pss_total, 2),
                    "native_heap_mb": round(native_heap, 2),
                    "java_heap_mb": round(java_heap, 2),
                    "battery_temp_c": round(temp_c, 1)
                }
            except Exception:
                pass
            time.sleep(1.0)

    def save_raw_event(self, event_type, payload):
        tier_dir = RAW_DIR / self.ram_tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.device_info.get('model', 'device')}_{event_type.lower()}_{ts_str}.json"
        target_path = tier_dir / filename

        full_event = {
            "device_profile": self.device_info,
            "hardware_snapshot": self.latest_hardware_snapshot,
            "telemetry_data": payload
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(full_event, f, indent=2)

        return target_path

    def append_ingestion_csv(self, payload):
        csv_path = PROCESSED_DIR / "ingestion_benchmarks.csv"
        file_exists = csv_path.exists()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        headers = [
            "timestamp", "ram_tier", "device_model", "soc", "doc_name", "file_size_kb",
            "page_count", "chunk_count", "embedding_model", "extraction_ms", "chunking_ms",
            "embedding_ms", "db_insert_ms", "total_ingest_ms", "pages_per_sec",
            "chunks_per_sec", "peak_pss_mb", "native_heap_mb"
        ]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([
                datetime.now().isoformat(),
                self.ram_tier,
                self.device_info.get("model"),
                self.device_info.get("soc_chipset"),
                payload.get("document_name"),
                round(payload.get("file_size_bytes", 0) / 1024.0, 2),
                payload.get("page_count"),
                payload.get("chunk_count"),
                payload.get("embedding_model_id"),
                payload.get("extraction_ms"),
                payload.get("chunking_ms"),
                payload.get("embedding_ms"),
                payload.get("db_insert_ms"),
                payload.get("total_ingest_ms"),
                round(payload.get("throughput_pages_per_sec", 0.0), 2),
                round(payload.get("throughput_chunks_per_sec", 0.0), 2),
                self.latest_hardware_snapshot.get("pss_total_mb"),
                self.latest_hardware_snapshot.get("native_heap_mb")
            ])

    def append_retrieval_csv(self, payload):
        csv_path = PROCESSED_DIR / "retrieval_benchmarks.csv"
        file_exists = csv_path.exists()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        headers = [
            "timestamp", "ram_tier", "device_model", "query", "intent", "total_corpus_chunks",
            "top_k_candidates", "filtered_chunks", "embedding_model", "intent_routing_ms",
            "query_embedding_ms", "dense_search_ms", "bm25_search_ms", "rrf_fusion_ms",
            "context_assembly_ms", "total_retrieval_ms", "assembled_tokens", "budget_tokens"
        ]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([
                datetime.now().isoformat(),
                self.ram_tier,
                self.device_info.get("model"),
                payload.get("query"),
                payload.get("intent"),
                payload.get("total_corpus_chunks"),
                payload.get("top_k_candidates"),
                payload.get("filtered_chunks"),
                payload.get("embedding_model_id"),
                payload.get("intent_routing_ms"),
                payload.get("query_embedding_ms"),
                payload.get("dense_search_ms"),
                payload.get("bm25_search_ms"),
                payload.get("rrf_fusion_ms"),
                payload.get("context_assembly_ms"),
                payload.get("total_retrieval_ms"),
                payload.get("assembled_tokens"),
                payload.get("budget_tokens")
            ])

    def append_generation_csv(self, payload):
        csv_path = PROCESSED_DIR / "generation_benchmarks.csv"
        file_exists = csv_path.exists()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        headers = [
            "timestamp", "ram_tier", "device_model", "soc", "model_name", "context_size",
            "prompt_tokens", "generated_tokens", "ttft_ms", "decode_ms",
            "total_generation_ms", "tokens_per_second", "peak_pss_mb", "native_heap_mb", "temp_c"
        ]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([
                datetime.now().isoformat(),
                self.ram_tier,
                self.device_info.get("model"),
                self.device_info.get("soc_chipset"),
                payload.get("model_name"),
                payload.get("context_size"),
                payload.get("prompt_tokens"),
                payload.get("generated_tokens"),
                payload.get("ttft_ms"),
                payload.get("decode_ms"),
                payload.get("total_generation_ms"),
                round(payload.get("tokens_per_second", 0.0), 2),
                self.latest_hardware_snapshot.get("pss_total_mb"),
                self.latest_hardware_snapshot.get("native_heap_mb"),
                self.latest_hardware_snapshot.get("battery_temp_c")
            ])

    def start_recording(self):
        if not self.detect_device():
            return

        # Start hardware poller thread
        poller_thread = threading.Thread(target=self.poll_hardware_stats, daemon=True)
        poller_thread.start()

        # Clear logcat buffer
        self.adb_cmd(["logcat", "-c"])

        print(f"\n[+] LISTENING FOR TELEMETRY EVENTS FROM '{PACKAGE_NAME}'...")
        print(f"[*] Raw logs will be saved to: benchmarks/data/raw/{self.ram_tier}/")
        print(f"[*] Consolidated tables will update in: benchmarks/data/processed/")
        print("[*] Press Ctrl+C at any time to stop recording.\n")

        adb_base = ["adb"]
        if self.serial:
            adb_base.extend(["-s", self.serial])
        adb_base.extend(["logcat", "-v", "raw", "-s", f"{BENCHMARK_TAG}:I"])

        proc = subprocess.Popen(adb_base, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

        try:
            for line in proc.stdout:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                    event_type = payload.get("event_type", "UNKNOWN")

                    raw_file = self.save_raw_event(event_type, payload)

                    if event_type == "DOCUMENT_INGESTION":
                        self.append_ingestion_csv(payload)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📄 INGESTION COMPLETE:")
                        print(f"    • Document: {payload.get('document_name')} ({payload.get('page_count')} pages, {payload.get('chunk_count')} chunks)")
                        print(f"    • Total Ingest Time: {payload.get('total_ingest_ms')} ms ({payload.get('throughput_pages_per_sec'):.2f} pages/sec)")
                        print(f"    • Saved: {raw_file.name}")

                    elif event_type == "RAG_RETRIEVAL":
                        self.append_retrieval_csv(payload)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 RETRIEVAL COMPLETE:")
                        print(f"    • Query: \"{payload.get('query')}\" (Intent: {payload.get('intent')})")
                        print(f"    • Total Retrieval: {payload.get('total_retrieval_ms')} ms | Routing: {payload.get('intent_routing_ms')} ms | Embed: {payload.get('query_embedding_ms')} ms | Search: {payload.get('dense_search_ms')} ms")
                        print(f"    • Assembled Context: {payload.get('assembled_tokens')} tokens")
                        print(f"    • Saved: {raw_file.name}")

                    elif event_type == "LLM_GENERATION":
                        self.append_generation_csv(payload)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 GENERATION COMPLETE:")
                        print(f"    • Model: {payload.get('model_name')}")
                        print(f"    • TTFT (Time To First Token): {payload.get('ttft_ms')} ms")
                        print(f"    • Throughput (Decode Speed): {payload.get('tokens_per_second'):.2f} tokens/sec ({payload.get('generated_tokens')} tokens in {payload.get('decode_ms')} ms)")
                        print(f"    • Peak RSS Memory: {self.latest_hardware_snapshot.get('pss_total_mb')} MB (Native Heap: {self.latest_hardware_snapshot.get('native_heap_mb')} MB)")
                        print(f"    • Saved: {raw_file.name}")

                    print("-" * 60)

                except json.JSONDecodeError:
                    continue
        except KeyboardInterrupt:
            print("\n[!] Recording stopped by user.")
        finally:
            self.stop_signal.set()
            proc.terminate()
            print("[+] Benchmarking session closed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OfflineDocAI USB Benchmark Recorder")
    parser.add_argument("-s", "--serial", help="Target ADB device serial (optional, auto-detected if omitted)")
    parser.add_argument("-t", "--tier", choices=["4gb", "6gb", "8gb"], help="Force RAM tier (tier_4gb, tier_6gb, tier_8gb). Auto-detected if omitted.")
    args = parser.parse_args()

    recorder = BenchmarkRecorder(serial=args.serial)
    if args.tier:
        recorder.ram_tier = f"tier_{args.tier.lower()}"
    recorder.start_recording()
