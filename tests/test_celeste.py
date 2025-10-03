from celeste import LLM, Grammar, CARS, MCMC

llm = LLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Create a simple grammar
grammar_str = """
start: "hello" | "world"
"""

grammar = Grammar.from_string(grammar_str, llm.tokenizer)

# Test CARS
print("Testing CARS...")
sampler = CARS(llm, grammar, max_new_tokens=10)
results = sampler.sample("Generate:", n_samples=2, n_steps=50)

for i, result in enumerate(results):
    print(f"Sample {i}: {result.text}")
    print(f"Log prob: {result.raw_logprob:.2f}")

# Test MCMC
print("\nTesting MCMC...")
mcmc_sampler = MCMC(llm, grammar, variant="prefix", max_new_tokens=10)
mcmc_results = mcmc_sampler.sample("Generate:", n_samples=2, n_steps=3)

for i, result in enumerate(mcmc_results):
    print(f"Sample {i}: {result.text}")
    print(f"Log prob: {result.raw_logprob:.2f}")

print("\nTests completed!")