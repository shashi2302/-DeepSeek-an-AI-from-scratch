import random
import math

# Corpus and vocab
corpus = [
    "the cat sat",
    "the dog ran",
    "the cat ran",
    "the dog sat"
]
vocab = sorted(set(" ".join(corpus).split()))  # ['cat', 'dog', 'ran', 'sat', 'the']
vocab_size = len(vocab)  # 5
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# Embeddings
embedding_size = 4
embeddings = {word: [random.uniform(-0.1, 0.1) for _ in range(embedding_size)] 
              for word in vocab}

# Network setup
layer_sizes = [embedding_size, 6, 6]  # Input -> Core1 -> Core2
weights = []
biases = []
memory_weights = []
for i in range(len(layer_sizes) - 1):
    w = [[random.uniform(-0.1, 0.1) for _ in range(layer_sizes[i + 1])] 
         for _ in range(layer_sizes[i])]
    weights.append(w)
    b = [random.uniform(-0.1, 0.1) for _ in range(layer_sizes[i + 1])]
    biases.append(b)
    mw = [[random.uniform(-0.05, 0.05) for _ in range(layer_sizes[i + 1])] 
          for _ in range(layer_sizes[i + 1])]
    memory_weights.append(mw)

# Output layer
output_weights = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] 
                  for _ in range(layer_sizes[-1])]
output_bias = [random.uniform(-0.1, 0.1) for _ in range(vocab_size)]

# Depth Core function
def depth_core_layer(inputs, w, b, mw):
    outputs = []
    for i in range(len(w[0])):
        x = sum(inputs[j] * w[j][i] for j in range(len(inputs))) + b[i]
        y = math.tanh(x)
        outputs.append(y)
    feedback = [sum(outputs[j] * mw[j][i] for j in range(len(outputs))) 
                for i in range(len(outputs))]
    return [o + 0.3 * math.tanh(f) for o, f in zip(outputs, feedback)]

# Output layer function
def softmax(x):
    exp_x = [math.exp(xi - max(x)) for xi in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]

def insight_forge(inputs, w, b):
    outputs = [sum(inputs[j] * w[j][i] for j in range(len(inputs))) + b[i] 
               for i in range(len(w[0]))]
    return softmax(outputs)

# Forward pass
def forward_pass(sentence):
    words = sentence.split()
    input_vector = embeddings[words[-1]]  # Last word as context
    layer_output = input_vector
    for w, b, mw in zip(weights, biases, memory_weights):
        layer_output = depth_core_layer(layer_output, w, b, mw)
    return insight_forge(layer_output, output_weights, output_bias)

# Loss and helpers
def cross_entropy(predicted, actual):
    return -sum(a * math.log(max(p, 1e-10)) for a, p in zip(actual, predicted))

def one_hot(index, size):
    return [1 if i == index else 0 for i in range(size)]

# Prediction function
def predict_next_word(sentence):
    probs = forward_pass(sentence)
    best_idx = probs.index(max(probs))
    return idx_to_word[best_idx]

# Training loop with progress
learning_rate = 0.01
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for sentence in corpus:
        words = sentence.split()
        input_sentence = " ".join(words[:-1])
        target_word = words[-1]
        target = one_hot(word_to_idx[target_word], vocab_size)
        
        predicted = forward_pass(input_sentence)
        loss = cross_entropy(predicted, target)
        total_loss += loss
        
        # Gradient update for weights
        for layer in range(len(weights)):
            for i in range(len(weights[layer])):
                for j in range(len(weights[layer][i])):
                    h = 0.001
                    original = weights[layer][i][j]
                    weights[layer][i][j] += h
                    new_pred = forward_pass(input_sentence)
                    new_loss = cross_entropy(new_pred, target)
                    gradient = (new_loss - loss) / h
                    weights[layer][i][j] = original - learning_rate * gradient
        
        # Gradient update for output weights
        for i in range(len(output_weights)):
            for j in range(len(output_weights[i])):
                h = 0.001
                original = output_weights[i][j]
                output_weights[i][j] += h
                new_pred = forward_pass(input_sentence)
                new_loss = cross_entropy(new_pred, target)
                gradient = (new_loss - loss) / h
                output_weights[i][j] = original - learning_rate * gradient
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(corpus):.4f}")

# Test 
test_sentences = ["the cat", "the dog"]
print("\nTesting DeepSeek:")
for test in test_sentences:
    prediction = predict_next_word(test)
    print(f"Input: '{test}' -> Predicted: '{prediction}'")
