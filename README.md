# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.

Receipt images -> Convert images to Base64 data URLs -> 

In my solution, build_chain() creates a LangChain pipeline with the deepseek-v4-flash-vision-exp vision model. The prompt instructs the model to read a receipt image and return a JSON object containing the final payment after rounding and the original amount before discounts. 

Next in answer_quries(), every receipt images are converted into a Base64 data URL, and I made one multimodal input for each receipt. The inputs are passed into the chain using chain.batch(). For each response, the JSON output is parsed and the two required monetary fields are extracted. I used Python's Decimal type to sum up the amounts accurately. 

1. How much money did I spend in total for these bills?
It is calculated by summing up the amounts of JSON field "mount_paid_after_rounding"

2. How much would I have had to pay without the discount?
It is calculated by summing up the amounts of JSON field "amount_without_discounts"

And finally, answer_quries() returns a dictionary using the question strings required by the assignment.

Here is a visualisation of my solution:

Receipt images -> Convert images to Base64 data URLs -> Multimodal prompt + DeepSeek vision model -> JSON fields for each receipt -> Parse and sum monetary values using Python -> Return required JSON object