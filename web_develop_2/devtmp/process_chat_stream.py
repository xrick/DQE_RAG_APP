

async def process_chat_stream(message: str, search_action: int, search_threshold: float) -> AsyncGenerator[str, None]:
    start_time = time()
    internal_results_markdown = "nodata"
    web_search_results = None
    final_answer = ""

    try:
        # === Step 1 & 2: Internal Search and Yield ===
        logging.info("Step 1: Performing internal search...")
        _poses = []
        _dists = []
        chk_ifnodata = "ragsearch"
        internal_data_source = None

        # (Include your existing logic for search_action 1 and 2 here to populate _poses, _dists, chk_ifnodata)
        # Determine internal_data_source (dfObj or dfObj_aitrial) based on search_action
        if search_action == 1:
             _pos, _distances = search_similar_questions(faiss_retriever, message);
             # ... (rest of your thresholding logic for action 1) ...
             if not _poses: chk_ifnodata="nodata"
             else: internal_data_source = dfObj
        elif search_action == 2:
             _pos_qsrc, _distances_qsrc = search_similar_questions(faiss_retriever_qsrc, message);
             _pos_module, _distance_module = search_similar_questions(faiss_retriever_module,message);
             # ... (rest of your combining logic for action 2) ...
             if not _poses: chk_ifnodata="nodata"
             else: internal_data_source = dfObj_aitrial
        # Add handling if search_action is not 1 or 2 but requires internal data later?

        if chk_ifnodata != "nodata" and internal_data_source is not None:
            value_matrix = convert_df_to_list(internal_data_source, _poses)
            # Optional: Format value_matrix into a simple markdown table or list for immediate display
            # For simplicity, let's just send the raw list for now, or a placeholder message
            # You might want to create a simplified formatting function here
            internal_results_formatted = f"Found {len(_poses)} potentially relevant internal items." # Placeholder
            # Example formatting (can be more complex):
            # headers=["模块", "严重度", "问题现象描述", ...] # Subset of required_columns
            # internal_results_markdown = generate_markdown_table(headers=headers[:3], value_matrix=[row[:3] for row in value_matrix])


            internal_results_data = {
                "type": "internal_results",
                "content": internal_results_formatted # Or internal_results_markdown
            }
            yield json.dumps(internal_results_data) + "\n"
            logging.info("Step 2: Streamed internal search results.")
        else:
             internal_results_data = {"type": "internal_results", "content": "nodata"}
             yield json.dumps(internal_results_data) + "\n"
             logging.info("Step 2: Streamed internal search result (no data found).")


        # === Step 3: Initiate Concurrent Web Search ===
        web_search_task = None
        # Decide if web search is needed (e.g., always, or based on search_action, or if internal results are insufficient)
        # Let's assume for this example, we *always* do it if internal search happened or if action == 3
        should_web_search = (chk_ifnodata != "nodata") or (search_action == 3)

        if should_web_search:
             logging.info("Step 3: Initiating web search concurrently...")
             yield json.dumps({"type": "status", "message": "Performing web search..."}) + "\n"
             # Use asyncio.create_task to run it in the background
             web_search_task = asyncio.create_task(do_google_serper_search(query=message))
        else:
             logging.info("Step 3: Skipping web search based on conditions.")


        # === Step 4 & 5: Wait for Web Search and Generate/Stream Final Answer ===
        final_answer_data = {"type": "final_answer", "content": ""}

        if web_search_task:
            logging.info("Step 4: Waiting for web search completion...")
            raw_web_results = await web_search_task
            logging.info("Step 4: Web search completed.")
            # Format web results if needed (your existing logic)
            web_search_results_markdown = format_serper_results_to_markdown(serper_data=raw_web_results)

            # Prepare for final generation using LLM
            # Option A: Use existing CompositiveGoogleSerperSummarizer (needs modification potentially)
            # Option B: Create a new prompt/chain combining internal_results_markdown and web_search_results_markdown

            # --- Using a simplified direct LLM call for demonstration ---
            # You should replace this with your actual summarization/generation logic
            # Ensure your LLM call is async (e.g., llm.ainvoke or similar)
            logging.info("Step 5: Generating final combined answer...")
            yield json.dumps({"type": "status", "message": "Generating final answer..."}) + "\n"

            # Construct the prompt input
            # Get internal context string/markdown
            internal_context_for_llm = "No internal data found."
            if chk_ifnodata != "nodata" and internal_data_source is not None:
                 # Convert value_matrix or use internal_results_markdown
                 internal_context_for_llm = json.dumps(convert_df_to_list(internal_data_source, _poses)) # Example

            # Define the combined prompt (adjust as needed)
            combined_template = """
            User Query: {query}

            Internal Knowledge Base Results:
            {internal_context}

            Web Search Results:
            {web_context}

            Task: Based on the user query, internal results, and web search results, provide a comprehensive answer.
            First, summarize the key findings from internal data relevant to the query.
            Second, summarize the key findings from the web search relevant to the query.
            Finally, provide a synthesized answer combining insights from both sources. Use Markdown format.
            Answer in Simplified Chinese.
            """
            combined_prompt = PromptTemplate(input_variables=["query", "internal_context", "web_context"], template=combined_template)
            chain = combined_prompt | llm # Assuming your LLMInitializer sets up LangChain compatible llm

            # Use astream for streaming output from LLM if available
            final_answer_full = ""
            async for chunk in chain.astream({
                "query": message,
                "internal_context": internal_context_for_llm,
                "web_context": web_search_results_markdown
            }):
                # Assuming 'chunk' is a string or has a 'content' attribute based on your LLM setup
                content_chunk = chunk if isinstance(chunk, str) else getattr(chunk, 'content', '')
                if content_chunk:
                    final_answer_full += content_chunk
                    chunk_data = {"type": "final_answer_chunk", "content": content_chunk}
                    yield json.dumps(chunk_data) + "\n"

            # If astream is not available or you need the full response first:
            # final_answer_full = await chain.ainvoke({ ... })
            # final_answer_data["content"] = final_answer_full
            # yield json.dumps(final_answer_data) + "\n"

            logging.info("Step 5: Finished streaming final answer.")

        else:
            # Handle case where only internal search was done (or neither)
            if chk_ifnodata != "nodata":
                 final_answer_data["content"] = internal_results_formatted # Or a summary of it
                 yield json.dumps(final_answer_data) + "\n"
            else:
                 final_answer_data["content"] = "No relevant information found internally or via web search."
                 yield json.dumps(final_answer_data) + "\n"

    except Exception as e:
        logging.error(f"Error during chat processing: {e}", exc_info=True)
        error_data = {"type": "error", "message": f"An error occurred: {str(e)}"}
        yield json.dumps(error_data) + "\n"
    finally:
        end_time = time()
        total_time = end_time - start_time
        logging.info(f"Total processing time: {total_time}")
        done_data = {"type": "done", "computation_time": total_time}
        yield json.dumps(done_data) + "\n"

