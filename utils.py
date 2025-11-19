import time
import asyncio
import aiohttp
import numpy as np

WARMUP_REQUESTS = 5
ESTIMATE_SAMPLES = 3

async def wait_for_server(base_url, timeout=300):
    print(f"\nWaiting for vLLM at {base_url}...")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(f"{base_url}/v1/models") as resp:
                    if resp.status == 200:
                        print("Server is ready.")
                        return True
            except:
                pass
            await asyncio.sleep(2)
    raise TimeoutError("vLLM server did not start within timeout period")

def generate_random_prompt(tokenizer, num_tokens):
    vocab_size = tokenizer.vocab_size
    random_token_ids = np.random.randint(0, vocab_size, size=num_tokens)
    return tokenizer.decode(random_token_ids, skip_special_tokens=True)
