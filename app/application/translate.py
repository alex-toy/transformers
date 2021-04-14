def translate(sentence):
    output = evaluate(sentence).numpy()
    
    predicted_sentence = tokenizer_es.decode(
        [i for i in output if i < VOCAB_SIZE_ES-2]
    )
    
    print("Entrada: {}".format(sentence))
    print("Traducción predicha: {}".format(predicted_sentence))