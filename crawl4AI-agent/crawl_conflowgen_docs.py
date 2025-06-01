import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
from conflowgen_urls import CONFLOWGEN_URLS

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
embeddingModel = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        embeddings = embeddingModel.encode(text).tolist()
        return embeddings
        # response = await openai_client.embeddings.create(
        #     model="text-embedding-3-small",
        #     input=text
        # )
        # return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 768  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary (currently commented out for simplicity)
    # extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "conflowgen_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title="title",  # extracted['title'] can be used here if needed
        summary="summary",  # extracted['summary'] can be used here if needed
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_conflowgen_docs_urls() -> List[str]:
    """Return the list of ConflowGen documentation URLs for crawling."""
    return ['https://conflowgen.readthedocs.io/latest/', 'https://conflowgen.readthedocs.io/latest/notebooks/first_steps.html', 'https://conflowgen.readthedocs.io/latest/notebooks/analyses.html', 'https://conflowgen.readthedocs.io/latest/contributing.html', 'https://conflowgen.readthedocs.io/latest/notebooks/in_spotlight.html', 'https://conflowgen.readthedocs.io/latest/notebooks/input_distributions.html', 'https://conflowgen.readthedocs.io/latest/background.html', 'https://conflowgen.readthedocs.io/latest/notebooks/previews.html', 'https://conflowgen.readthedocs.io/latest/api.html', 'https://conflowgen.readthedocs.io/latest/imprint.html', 'https://conflowgen.readthedocs.io/latest/bibliography.html', 'https://conflowgen.readthedocs.io/latest/examples.html', 'https://conflowgen.readthedocs.io/latest/notebooks/first_steps.ipynb', 'https://conflowgen.readthedocs.io/latest/index.html', 'https://conflowgen.readthedocs.io/latest/notebooks/analyses.ipynb', 'https://conflowgen.readthedocs.io/latest/notebooks/in_spotlight.ipynb', 'https://conflowgen.readthedocs.io/latest/notebooks/input_distributions.ipynb', 'https://conflowgen.readthedocs.io/latest/notebooks/previews.ipynb', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/database_chooser.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/modal_split_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/data_summaries/data_summaries_cache.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_by_vehicle_instance_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/outbound_to_inbound_vehicle_capacity_utilization_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/mode_of_transport_distribution_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/truck_gate_throughput_preview_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/descriptive_datatypes.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/truck_gate_throughput_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/domain_models/data_types/mode_of_transport.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/container_flow_by_vehicle_type_preview.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/modal_split_preview.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_adjustment_by_vehicle_type_analysis_summary.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/truck_gate_throughput_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/inbound_and_outbound_vehicle_capacity_preview.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/modal_split_preview_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/yard_capacity_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/container_dwell_time_distribution_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/quay_side_throughput_preview.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_adjustment_by_vehicle_type_analysis_summary_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/tools/continuous_distribution.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/modal_split_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/quay_side_throughput_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/reporting/output_style.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/domain_models/data_types/container_length.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/yard_capacity_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/container_flow_generation_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/port_call_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/outbound_to_inbound_vehicle_capacity_utilization_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/truck_arrival_distribution_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/export_container_flow_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_by_vehicle_type_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/quay_side_throughput_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_dwell_time_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/inbound_and_outbound_vehicle_capacity_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/inbound_and_outbound_vehicle_capacity_preview_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/vehicle_capacity_exceeded_preview.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/quay_side_throughput_preview_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/container_flow_by_vehicle_type_preview_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/vehicle_capacity_exceeded_preview_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/logger/logger.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_by_vehicle_instance_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/inbound_and_outbound_vehicle_capacity_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/container_length_distribution_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_by_vehicle_type_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_vehicle_type_adjustment_per_vehicle_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/storage_requirement_distribution_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/domain_models/data_types/storage_requirement.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews/truck_gate_throughput_preview.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_dwell_time_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/previews.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_adjustment_by_vehicle_type_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_vehicle_type_adjustment_per_vehicle_analysis_report.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/domain_models/distribution_models/container_dwell_time_distribution.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/analyses/container_flow_adjustment_by_vehicle_type_analysis.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/api/container_weight_distribution_manager.html', 'https://conflowgen.readthedocs.io/latest/_modules/conflowgen/application/data_types/export_file_format.html', 'https://conflowgen.readthedocs.io/latest/_modules/index.html']

async def main():
    urls = CONFLOWGEN_URLS
    if not urls:
        print("No URLs found to crawl")
        return

    print(f"Found {len(urls)} ConFlowGen documentation URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
