from transformers import pipeline

# Load the pre-trained question-answering model
qa_pipeline = pipeline("question-answering")

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

# Ask a question
question = "What is the Amazon rainforest often called?"

# Get the answer using the pre-trained model
answer = qa_pipeline(question=question, context=paragraph)

# Print the answer
print(f"Question: {question}")
print(f"Answer: {answer['answer']} (Confidence: {answer['score']:.4f})")
