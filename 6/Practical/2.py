INITIAL_PROBS = list(map(float, input().split()))
TRANSITION_PROBS = [list(map(float, input().split())) for _ in range(3)]
EMISSION_PROBS = [list(map(float, input().split())) for _ in range(3)]
EVIDENCE = list(map(int, input().split()))

current_probs = INITIAL_PROBS.copy()
best_probs = []
for i in range(len(current_probs)):
    current_probs[i] *= EMISSION_PROBS[i][EVIDENCE[0]]
#print(current_probs)
best_probs.append(current_probs.index(max(current_probs)))
for evidence in EVIDENCE[1:]:
    new_probs = [0, 0, 0]
    for i in range(len(new_probs)):
        for (index, prob) in enumerate(current_probs): # forward
            new_probs[i] += prob * TRANSITION_PROBS[index][i]
        new_probs[i] *= EMISSION_PROBS[i][evidence]
    #print(new_probs)
    current_probs = new_probs
    best_probs.append(current_probs.index(max(current_probs)))

def num_to_temp(n: int) -> str:
    if n == 0:
        return "cold"
    if n == 1:
        return "medium"
    if n == 2:
        return "hot"
    return "duck"
prob_sum = sum(current_probs)
current_probs = list(map(lambda x: round(x / prob_sum, 3), current_probs)) # normalize
print("Most likely sequence of states:")
print(list(map(num_to_temp, best_probs)))
print("Probability of states after observing:")
print(current_probs)