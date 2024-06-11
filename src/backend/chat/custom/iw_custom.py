import logging
import json

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from fastapi import HTTPException

from backend.chat.collate import combine_documents
from backend.chat.custom.utils import get_deployment
from backend.config.tools import AVAILABLE_TOOLS, ToolName
from backend.model_deployments.base import BaseDeployment
from backend.schemas.cohere_chat import CohereChatRequest
from backend.schemas.chat import StreamToolInput
from backend.schemas.tool import Category, Tool
from backend.services.logger import get_logger

import pprint

class IWCustomChat:
    """Custom chat flow not using integrations for models."""

    logger = get_logger()


    def old_chunk_and_rerank(self, deployment_model, query, documents, max_chunk_size=300):
      # TODO(xjdr): This should be tokens not chars or words
      docs = []
      chunks = []
      out = []
      for document in documents:
        text = documents['text']
        splits = text.split('\n')
        docs.append(document)
        chunks.append(text)
      res = deployment_model.invoke_rerank(query=query, documents=chunks, top_n=4)
      res.results.sort(key=lambda x: x.relevance_score, reverse=True)
      for r in res.results:
        idx = r.index
        out.append(docs[idx])
      return out


    def chunk_and_rerank(self, deployment_model, query, documents, max_chunk_size=300):
      # TODO(xjdr): This should be tokens not chars or words
      chunks = []
      for document in documents:
        text = document['text']
        # should split on '. '
        words = text.split()  # Split into individual words
        current_chunk = []
        for word in words:
          if len(current_chunk) + len(word) > max_chunk_size:
            chunks.append(' '.join(current_chunk))  # Join words in chunk with spaces
            current_chunk = []
          current_chunk.append(word)
        chunks.append(' '.join(current_chunk))  # Add the last chunk
      res = deployment_model.invoke_rerank(query=query, documents=chunks, top_n=4)
      res.results.sort(key=lambda x: x.relevance_score, reverse=True)
      out = []
      for r in res.results:
        idx = r.index
        out.append({'text': chunks[idx]})
      return out

    def do_search_call(self, tool_call):
        all_documents = {}
        tool = AVAILABLE_TOOLS.get(tool_call.name)
        if tool is None:
            return all_documents
        if tool.category != Category.DataLoader:
            return all_documents
        retriever = tool.implementation(**tool.kwargs)
        if not retriever:
            return all_documents
        if 'query' not in tool_call.parameters:
            return all_documents
        query = tool_call.parameters['query']
        parameters = {"query": query}
        all_documents[query] = retriever.call(parameters)
        return all_documents

    def do_tool_call(self, tool_call):
        match tool_call.name:
            case 'Internet Search':
                return self.do_search_call(tool_call)
            case 'Python_Interpreter':
                tool = AVAILABLE_TOOLS.get(tool_call.name)
                interp_out = tool.implementation().call(
                    parameters=tool_call.parameters,
                )
                print(f'DEBUG Python_Interpreter {interp_out=}')
                return {'python_output': interp_out}
            case 'Wikipedia':
                return self.do_search_call(tool_call)
            case 'search_file':
                pass
            case 'read_document':
                pass
            case _:
                return None

    def eval_step(self, deployment_model, chat_request, tool_calls):
        for tool_call in tool_calls:
            # if tool == "DIRECTLY_ANSWER" call model here and return None, stream_events, False blindly
            print(f'DEBUG: {tool_call=}')
            document_updates = self.do_tool_call(tool_call)
            pretty_document_updates = pprint.pformat(document_updates)
            print(f'DEBUG: {pretty_document_updates}')
            if document_updates:
              #chat_request.documents.extend(combine_documents(documents=document_updates, model=deployment_model))
              #chat_request.documents = combine_documents(documents=document_updates, model=deployment_model)
                for k,v in document_updates.items():
                    all_docs = chat_request.documents + v
                    print(f'KJHKSJDF: {all_docs=}')
                    documents = self.chunk_and_rerank(deployment_model, k, all_docs)
                    #chat_request.documents.extend(documents)
                    print(f'KJHKSJDF: {documents=}')
                    chat_request.documents = documents
        new_tool_calls = []
        filtered_events = []
        all_events = []
        stream_events = deployment_model.invoke_chat_stream(chat_request)
        new_chat_request = CohereChatRequest(**chat_request.model_dump())
        prev_message = new_chat_request.message
        for event in stream_events:
            all_events.append(event)
            if event['event_type'] == 'tool-input':
                print(f'DEBUG: TOOL INPUT {event=}')
                prev_message += '\n\n' + event['text']
                filtered_events.append(event)
            elif event['event_type'] == 'tool-calls-generation':
                print(f'DEBUG: TOOL CALL {event=}')
                for tc in event['tool_calls']:
                    tcd = {
                        'tool_name': tc.name,
                        'parameters': tc.parameters,
                    }
                    prev_message += f'Action:\n{json.dumps(tcd)}'
                new_tool_calls.extend(event.get('tool_calls', []))
            elif event['event_type'] == 'text-generation':
                print(f'DEBUG: TEXT GEN {event=}')
                #new_chat_request.message = event['text']
            elif event['event_type'] == 'citation-generation':
                print(f'DEBUG: CITATION GEN {event=}')
                filtered_events.append(event)
            elif event['event_type'] == 'stream-end':
                print(f'DEBUG: STREAM END {event=}')
                return None, [], all_events, False
            else:
                # We should be better here
                print(f'DEBUG: UNKNOWN {event=}')
                filtered_events.append(event)
        #new_chat_request.stream = True
        #new_chat_request.tools = chat_request.tools
        # Single Turn Agent Only Baby
        new_chat_request.tools = []
        if "\nObservation:" in prev_message:
            new_chat_request.message = prev_message
        else:
            new_chat_request.message = prev_message + "\nObservation:\n"
        print(f'DEBUG: NEW REQ {new_chat_request=}')
        return new_chat_request, new_tool_calls, filtered_events, True

    def get_managed_tools(self, chat_request: CohereChatRequest) -> List[Any]:
        managed_tools = []
        if chat_request.tools:
            for tool in chat_request.tools:
                available_tool = AVAILABLE_TOOLS.get(tool.name)
                print(f'DEBUG {available_tool=}')
                if available_tool:
                    managed_tools.append(available_tool)
        return managed_tools

    def tool_time(self, chat_request: CohereChatRequest, deployment_model: Any, stream: bool, file_paths: Optional[List[str]]) -> Any:
      # Single Turn Agent Only for Now
      max_loop = 2
      cur_loop = 0
      cond = True
      print(f'DEBUG {chat_request.tools=}')
      chat_request.tools = self.get_managed_tools(chat_request)
      tool_calls = []
      while cond and cur_loop < max_loop:
        print("Step . step . steppin up! {cur_loop=}")
        chat_request, tool_calls, events, cond = self.eval_step(deployment_model, chat_request, tool_calls)
        for event in events:
          pretty_event = pprint.pformat(event)
          print(f'DEBUG: yielding filtered {pretty_event}')
          yield event
        cur_loop += 1

    def chat_time(self, chat_request: CohereChatRequest, deployment_model: Any, stream: bool):
        # Generate Response
        if stream:
            return deployment_model.invoke_chat_stream(chat_request)
        else:
            return deployment_model.invoke_chat(chat_request)

    def chat(self, chat_request: CohereChatRequest, 
             stream: bool,
             deployment_name: str,
             deployment_config: Dict[Any, Any],
             file_paths: Optional[List[str]],
             managed_tools: bool,
             **kwargs: Any) -> Any:
        """
        Chat flow for custom models.

        Args:
            chat_request (CohereChatRequest): Chat request.
            **kwargs (Any): Keyword arguments.

        Returns:
            Generator[StreamResponse, None, None]: Chat response.
        """
        # Choose the deployment model - validation already performed by request validator
        deployment_model = get_deployment(deployment_name)
        self.logger.info(f"Using deployment {deployment_model.__class__.__name__}")
        print(f'DEBUG CustomChat {chat_request=}')

        # def generator_func():
        #   for i in infinity_pool:
        #     yield message
        # return generator_func
        ## flow
        # 1. if managed_tools:
        #     managed_tools request for plan
        #     get back plan with tool calls
        #     make tool calls
        #     managed tools request for answer with tool call results
        # 1. else:
        #     chat request

        if managed_tools:
            return self.tool_time(chat_request, deployment_model, stream, file_paths)
        else:
            return self.chat_time(chat_request, deployment_model, stream)
        # if len(chat_request.tools) > 0 and len(chat_request.documents) > 0:
        #     raise HTTPException(
        #         status_code=400, detail="Both tools and documents cannot be provided."
        #     )
        #
        # if kwargs.get("managed_tools", True):
        #     # Generate Search Queries
        #     chat_history = [message.to_dict() for message in chat_request.chat_history]
        #
        #     function_tools: list[Tool] = []
        #     for tool in chat_request.tools:
        #         available_tool = AVAILABLE_TOOLS.get(tool.name)
        #         if available_tool and available_tool.category == Category.Function:
        #             function_tools.append(Tool(**available_tool.model_dump()))
        #
        #     if len(function_tools) > 0:
        #         print(f'DEBUG {function_tools=}')
        #         tool_results = self.get_tool_results(
        #             chat_request.message, function_tools, deployment_model
        #         )
        #
        #         chat_request.tools = None
        #         print('DEBUG returning chat results')
        #         if kwargs.get("stream", True) is True:
        #             return deployment_model.invoke_chat_stream(
        #                 chat_request,
        #                 tool_results=tool_results,
        #             )
        #         else:
        #             return deployment_model.invoke_chat(
        #                 chat_request,
        #                 tool_results=tool_results,
        #             )
        #
        #     print('DEBUG invoke_search_queries')
        #     queries = deployment_model.invoke_search_queries(
        #         chat_request.message, chat_history
        #     )
        #     self.logger.info(f"Search queries generated: {queries}")
        #
        #     # Fetch Documents
        #     retrievers = self.get_retrievers(
        #         kwargs.get("file_paths", []), [tool.name for tool in chat_request.tools]
        #     )
        #     self.logger.info(
        #         f"Using retrievers: {[retriever.__class__.__name__ for retriever in retrievers]}"
        #     )
        #
        #     # No search queries were generated but retrievers were selected, use user message as query
        #     if len(queries) == 0 and len(retrievers) > 0:
        #         queries = [chat_request.message]
        #
        #     all_documents = {}
        #     # TODO: call in parallel and error handling
        #     # TODO: merge with regular function tools after multihop implemented
        #     for retriever in retrievers:
        #         for query in queries:
        #             print(f'DEBUG {query=} {retriever=}')
        #             parameters = {"query": query}
        #             all_documents.setdefault(query, []).extend(
        #                 retriever.call(parameters)
        #             )
        #
        #     print(f'DEBUG {all_documents=}')
        #     # Collate Documents
        #     documents = combine_documents(all_documents, deployment_model)
        #     chat_request.documents = documents
        #     chat_request.tools = []
        #
        # # Generate Response
        # if kwargs.get("stream", True) is True:
        #     return deployment_model.invoke_chat_stream(chat_request)
        # else:
        #     return deployment_model.invoke_chat(chat_request)

    def get_retrievers(
        self, file_paths: list[str], req_tools: list[ToolName]
    ) -> list[Any]:
        """
        Get retrievers for the required tools.

        Args:
            file_paths (list[str]): File paths.
            req_tools (list[str]): Required tools.

        Returns:
            list[Any]: Retriever implementations.
        """
        retrievers = []

        # If no tools are required, return an empty list
        if not req_tools:
            return retrievers

        # Iterate through the required tools and check if they are available
        # If so, add the implementation to the list of retrievers
        # If not, raise an HTTPException
        for tool_name in req_tools:
            tool = AVAILABLE_TOOLS.get(tool_name)
            if tool is None:
                raise HTTPException(
                    status_code=404, detail=f"Tool {tool_name} not found."
                )

            # Check if the tool is visible, if not, skip it
            if not tool.is_visible:
                continue

            if tool.category == Category.FileLoader and file_paths is not None:
                for file_path in file_paths:
                    retrievers.append(tool.implementation(file_path, **tool.kwargs))
            elif tool.category != Category.FileLoader:
                retrievers.append(tool.implementation(**tool.kwargs))

        return retrievers

    def get_tool_results(
        self, message: str, tools: list[Tool], model: BaseDeployment
    ) -> list[dict[str, Any]]:
        tool_results = []
        tools_to_use = model.invoke_tools(message, tools)

        print(f'DEBUG {tools_to_use}')
        tool_calls = tools_to_use.tool_calls if tools_to_use.tool_calls else []
        for tool_call in tool_calls:
            tool = AVAILABLE_TOOLS.get(tool_call.name)

            if not tool:
                logging.warning(f"Couldn't find tool {tool_call.name}")
                continue

            outputs = tool.implementation().call(
                parameters=tool_call.parameters,
            )

            # If the tool returns a list of outputs, append each output to the tool_results list
            # Otherwise, append the single output to the tool_results list
            outputs = outputs if isinstance(outputs, list) else [outputs]
            for output in outputs:
                tool_results.append({"call": tool_call, "outputs": [output]})

        return tool_results
