
# Import the required libraries
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Sample paragraph
paragraph = """
The Amazon rainforest, also known as the Amazon jungle, is a large tropical rainforest situated in the Amazon basin
of South America. It covers an area of about 7 million square kilometers and is home to an incredibly diverse
ecosystem. The rainforest is often referred to as the "lungs of the Earth" because it produces a significant
amount of the world's oxygen.

The Amazon rainforest is facing various threats, including deforestation, illegal logging, and climate change.
These activities are causing a loss of biodiversity and impacting the indigenous communities that call the
rainforest home. Conservation efforts are crucial to preserving this vital ecosystem and addressing the
environmental challenges it faces.
"""

# Function to initialize model and tokenizer with the paragraph
def initialize_model(paragraph):
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer(paragraph, return_tensors="pt")
    outputs = model(**inputs)  # Run the model on the paragraph to cache some computations

    return model, tokenizer

# Function to ask a question using a given model and tokenizer
def ask_question(question, model, tokenizer):
    inputs = tokenizer(question, paragraph, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Initialize model and tokenizer with the paragraph
qa_model, qa_tokenizer = initialize_model(paragraph)

# Sample questions
questions = ["What is the Amazon rainforest often called?", "What threats does the Amazon rainforest face?"]

# Ask questions without reloading the paragraph each time
for q in questions:
    answer = ask_question(q, qa_model, qa_tokenizer)
    print(f"Question: {q}")
    print(f"Answer: {answer}")
    print()
