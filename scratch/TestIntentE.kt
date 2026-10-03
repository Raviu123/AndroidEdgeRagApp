import com.siddu.offlinedocai.ai.retrieval.*
fun main() {
    val q1 = "Summarize pages 1 to 3."
    println(QueryIntentRouter.route(q1).intent)
}
