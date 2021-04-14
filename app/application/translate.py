from app.application.evaluate import evaluate

def translate(sentence):
    output = evaluate(sentence).numpy()
    
    predicted_sentence = tokenizer_es.decode(
        [i for i in output if i < VOCAB_SIZE_ES-2]
    )
    
    print("Entry: {}".format(sentence))
    print("Predicted translation : {}".format(predicted_sentence))