

@app.post("/api/ai-chat-stream")
async def api_ai_chat_stream(request: Request):
    """
    Handles chat requests with streaming response.
    Streams internal results first, then performs web search concurrently,
    and finally streams the combined generated answer.
    """
    try:
        data = await request.json()
        message = data.get("message")
        search_action = int(data.get("search_action", 1)) # Default to precise
        search_threshold = float(data.get("search_threshold", 25.0))

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Return the streaming response, using the async generator
        # Use application/x-ndjson (newline delimited json) for easy frontend parsing
        return StreamingResponse(
            process_chat_stream(message, search_action, search_threshold),
            media_type="application/x-ndjson" # Newline Delimited JSON
        )
    except Exception as e:
        logging.error(f"Error in stream endpoint setup: {e}", exc_info=True)
        # StreamingResponse can't easily return HTTP errors after starting,
        # so initial validation errors are raised as HTTPException.
        # Errors during generation are handled within the generator.
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")

