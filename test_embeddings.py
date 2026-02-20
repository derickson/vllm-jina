#!/usr/bin/env python3
"""Smoke test for the vLLM Jina Embeddings v5 Nano service."""

import asyncio
import json
import os
import random
import sys
import time

import aiohttp
import requests as _requests

MODEL = "jinaai/jina-embeddings-v5-text-nano-retrieval"
BASE_URL = "http://0.0.0.0:8333/v1/embeddings"

NUM_WORKERS = 4
NUM_BATCHES = 1000
BATCH_SIZE = 100
TEXTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_texts.txt")


def post(payload: dict) -> dict:
    resp = _requests.post(BASE_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


async def async_post(session: aiohttp.ClientSession, payload: dict) -> dict:
    async with session.post(BASE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        resp.raise_for_status()
        return await resp.json()


def test_basic_embedding():
    print("Test 1: Basic embedding (full 768 dims)...")
    payload = {"model": MODEL, "input": "Hello, world!"}
    result = post(payload)
    embedding = result["data"][0]["embedding"]
    assert len(embedding) == 768, f"Expected 768 dims, got {len(embedding)}"
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print("  PASSED")


def test_matryoshka_truncation():
    print("Test 2: Matryoshka truncation to 128 dims...")
    payload = {
        "model": MODEL,
        "input": "Matryoshka embeddings support variable dimensions.",
        "dimensions": 128,
    }
    result = post(payload)
    embedding = result["data"][0]["embedding"]
    assert len(embedding) == 128, f"Expected 128 dims, got {len(embedding)}"
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print("  PASSED")


def test_batch_input():
    print("Test 3: Batch input (multiple strings)...")
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    payload = {"model": MODEL, "input": texts}
    result = post(payload)
    assert len(result["data"]) == 3, f"Expected 3 embeddings, got {len(result['data'])}"
    for i, item in enumerate(result["data"]):
        assert len(item["embedding"]) == 768, f"Item {i}: expected 768 dims"
    print(f"  Received {len(result['data'])} embeddings, each 768-dim")
    print("  PASSED")


WORDS = [
    "apple", "banana", "castle", "dragon", "elephant", "forest", "guitar",
    "horizon", "island", "jungle", "kitchen", "lantern", "mountain", "nebula",
    "ocean", "pyramid", "quantum", "river", "sunset", "tornado", "umbrella",
    "volcano", "whisper", "xylophone", "yellow", "zenith", "anchor", "breeze",
    "crystal", "dolphin", "ember", "falcon", "glacier", "harbor", "ivory",
    "jasmine", "kaleidoscope", "labyrinth", "marble", "nucleus", "orchid",
    "phantom", "quartz", "riddle", "sapphire", "thunder", "utopia", "vortex",
    "willow", "zeppelin", "alchemy", "blizzard", "canopy", "dagger", "eclipse",
    "furnace", "garnet", "hammock", "ignite", "jubilee", "keystone", "limestone",
    "monsoon", "nomad", "obelisk", "paragon", "quiver", "rampart", "scaffold",
    "talisman", "undertow", "vanguard", "warehouse", "xenon", "yeoman", "zodiac",
    "abacus", "bastion", "chimera", "drizzle", "enigma", "fjord", "gazelle",
    "hemisphere", "inferno", "jigsaw", "kinetic", "longitude", "mirage", "nautical",
    "obsidian", "pendulum", "quicksand", "resonance", "stalactite", "tributary",
    "ultraviolet", "vermillion", "wavelength", "xylem", "yonder", "zephyr",
    "almanac", "boulevard", "cascade", "dominion", "equinox", "fortress", "gondola",
    "hologram", "impulse", "javelin", "knapsack", "latitude", "meridian", "nexus",
    "overture", "pinnacle", "quarantine", "requiem", "spectrum", "threshold",
    "universe", "velocity", "wraith", "xerograph", "yearling", "zigzag",
    "artifact", "bramble", "catalyst", "debris", "ephemeral", "fractal", "geyser",
    "hybrid", "isotope", "juxtapose", "kerosene", "luminous", "monolith", "navigate",
    "optimize", "paradox", "quintuple", "renegade", "solstice", "tempest",
    "unravel", "vigilant", "wisteria", "xenolith", "yucca", "zirconia",
    "abyss", "binary", "conduit", "detour", "electrode", "filament", "graphite",
    "hexagon", "incandescent", "juncture", "krypton", "lattice", "membrane",
    "nocturnal", "osmosis", "phosphor", "quasar", "refraction", "synthesis",
    "tangent", "umbra", "vertex", "waveform", "xeric", "yielding", "zeolite",
    "aerosol", "biome", "cylinder", "delta", "erosion", "flux", "gradient",
    "hydraulic", "inertia", "junction", "kilowatt", "leverage", "momentum",
    "neutron", "orbital", "plasma", "quarry", "reactor", "substrate", "thermal",
    "uplift", "vacuum", "windmill", "xenotime", "yardarm", "zambia",
    "alloy", "beacon", "cobalt", "dynamo", "entropy", "fission", "galvanic",
    "helium", "integral", "jetstream", "kelvin", "lithium", "magnetar", "nozzle",
    "optic", "proton", "quench", "rotor", "solenoid", "turbine", "uranium",
    "viscosity", "weld", "xerafin", "yttrium", "zinc", "amplitude", "barometer",
    "capacitor", "diode", "emitter", "frequency", "gauge", "hertz", "inductor",
    "joule", "kilohertz", "lumen", "microfarad", "nanotube", "ohm", "photon",
    "qubit", "resistor", "semiconductor", "transistor", "ultrasonic", "voltmeter",
    "wattage", "xerographic", "yottabyte", "zettabyte", "algorithm", "bytecode",
    "compiler", "debugger", "executable", "firmware", "gateway", "hashmap",
    "iterator", "javascript", "kernel", "linker", "middleware", "namespace",
    "overflow", "pipeline", "querystring", "runtime", "servlet", "tokenizer",
    "unicode", "validator", "webhook", "xmlparser", "yacc", "zipfile",
    "abstraction", "benchmark", "coroutine", "daemon", "endpoint", "firewall",
    "goroutine", "handshake", "interface", "jsonpath", "kubernetes", "loadbalancer",
    "mutex", "nginx", "orchestrator", "postgres", "queue", "redis", "scheduler",
    "terraform", "upstream", "virtualenv", "websocket", "xpath", "yaml", "zookeeper",
    "aggregate", "bitfield", "callback", "decorator", "enumeration", "factory",
    "generator", "histogram", "injection", "journal", "keyframe", "listener",
    "marshaller", "notifier", "observer", "predicate", "quantizer", "resolver",
    "serializer", "transformer", "unmarshaller", "visitor", "wrapper", "xorshift",
    "yielder", "zipper", "accessor", "buffer", "channel", "dispatcher", "emitter",
    "formatter", "groupby", "handler", "indexer", "joiner", "keyring", "logger",
    "mapper", "normalizer", "operator", "parser", "qualifier", "reducer",
    "splitter", "tracker", "unpacker", "verifier", "watcher", "xfader",
    "yielding", "zeroing", "allocator", "batcher", "cacher", "deduplicator",
    "encoder", "fetcher", "granter", "hasher", "interpolator", "jitter",
    "keeper", "limiter", "merger", "negotiator", "optimizer", "profiler",
    "quantifier", "replicator", "sampler", "throttler", "uploader", "validator",
    "weigher", "extractor", "yielder", "compressor",
]


def random_text(num_words=256) -> str:
    return " ".join(random.choices(WORDS, k=num_words))


def generate_texts_file():
    total = NUM_BATCHES * BATCH_SIZE
    print(f"Generating {total} texts ({NUM_BATCHES} batches x {BATCH_SIZE}) to {TEXTS_FILE}...")
    with open(TEXTS_FILE, "w") as f:
        for i in range(total):
            f.write(random_text() + "\n")
    print(f"  Written {total} lines to {TEXTS_FILE}")


def load_texts() -> list[str]:
    if not os.path.exists(TEXTS_FILE):
        print(f"  Texts file not found, generating: {TEXTS_FILE}")
        generate_texts_file()
    with open(TEXTS_FILE, "r") as f:
        texts = [line.rstrip("\n") for line in f]
    expected = NUM_BATCHES * BATCH_SIZE
    assert len(texts) == expected, (
        f"Expected {expected} lines in {TEXTS_FILE}, got {len(texts)}. "
        f"Re-run with --generate to recreate."
    )
    return texts


async def _worker(worker_id: int, queue: asyncio.Queue, session: aiohttp.ClientSession,
                  all_texts: list[str], stats: dict):
    while True:
        try:
            batch_idx = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        start = batch_idx * BATCH_SIZE
        texts = all_texts[start : start + BATCH_SIZE]
        payload = {"model": MODEL, "input": texts}
        result = await async_post(session, payload)
        assert len(result["data"]) == BATCH_SIZE, (
            f"Batch {batch_idx}: expected {BATCH_SIZE} embeddings, got {len(result['data'])}"
        )
        for i, item in enumerate(result["data"]):
            assert len(item["embedding"]) == 768, (
                f"Batch {batch_idx}, item {i}: expected 768 dims"
            )
        stats["completed"] += 1
        stats["total_embeddings"] += BATCH_SIZE
        completed = stats["completed"]
        if completed % 100 == 0 or completed == 1:
            elapsed = time.perf_counter() - stats["t0"]
            eps = stats["total_embeddings"] / elapsed
            print(f"  Batch {completed}/{stats['num_batches']}: OK ({BATCH_SIZE} x 768-dim) | {eps:.1f} embeddings/sec")


async def _run_async_batches() -> dict:
    print(f"Test 4: {NUM_BATCHES} batches of {BATCH_SIZE} embeddings (~256 words each), {NUM_WORKERS} async workers...")
    print(f"  Loading texts from {TEXTS_FILE}...")
    all_texts = load_texts()
    print(f"  Loaded {len(all_texts)} texts")

    queue = asyncio.Queue()
    for i in range(NUM_BATCHES):
        queue.put_nowait(i)

    stats = {"completed": 0, "total_embeddings": 0, "t0": time.perf_counter(), "num_batches": NUM_BATCHES}

    async with aiohttp.ClientSession() as session:
        workers = [_worker(w, queue, session, all_texts, stats) for w in range(NUM_WORKERS)]
        await asyncio.gather(*workers)

    elapsed = time.perf_counter() - stats["t0"]
    eps = stats["total_embeddings"] / elapsed
    print(f"  Total: {stats['total_embeddings']} embeddings in {elapsed:.1f}s = {eps:.1f} embeddings/sec")
    print("  PASSED")
    return {"total_embeddings": stats["total_embeddings"], "elapsed": elapsed}


def test_async_batches() -> dict:
    return asyncio.run(_run_async_batches())


def main():
    print(f"Testing vLLM embedding service at {BASE_URL}")
    print(f"Model: {MODEL}")
    print()

    failures = []
    batch_stats = None
    t_start = time.perf_counter()

    for test_fn in [test_basic_embedding, test_matryoshka_truncation, test_batch_input]:
        try:
            test_fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append(test_fn.__name__)
        print()

    try:
        batch_stats = test_async_batches()
    except Exception as e:
        print(f"  FAILED: {e}")
        failures.append("test_async_batches")
    print()

    t_total = time.perf_counter() - t_start

    # --- Statistics summary ---
    print("=" * 60)
    print("  TEST STATISTICS")
    print("=" * 60)
    print(f"  Model:              {MODEL}")
    print(f"  Endpoint:           {BASE_URL}")
    print(f"  Async workers:      {NUM_WORKERS}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Number of batches:  {NUM_BATCHES}")
    print(f"  Total texts:        {NUM_BATCHES * BATCH_SIZE:,}")
    print(f"  Texts file:         {TEXTS_FILE}")
    print(f"  Total wall time:    {t_total:.1f}s")
    if batch_stats:
        eps = batch_stats["total_embeddings"] / batch_stats["elapsed"]
        print(f"  Batch test time:    {batch_stats['elapsed']:.1f}s")
        print(f"  Embeddings/sec:     {eps:,.1f}")
    print(f"  Tests passed:       {4 - len(failures)}/4")
    if failures:
        print(f"  Failed:             {', '.join(failures)}")
    print("=" * 60)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_texts_file()
    else:
        main()
