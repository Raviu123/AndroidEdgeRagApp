fun main() {
    val query = "Where in the document is the battery discussed?"
    val intent = com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter.route(query).intent
    println("Intent: $intent")
}
