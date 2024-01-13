from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

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

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
inputs = tokenizer(paragraph, return_tensors="pt")
outputs = model(**inputs)

@api_view(['GET'])
def get_answer(request):
    question = request.query_params.get('question', '')
    inputs = tokenizer(question, paragraph, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return Response({'question': question, 'answer': answer})