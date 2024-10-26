import json
import random


random.seed(42)


def load_sound_mappings(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_syllable_count(num_polylines):
    if num_polylines < 200:
        return 2
    elif num_polylines <= 500:
        return 3
    else:
        return 4


def select_sounds(
    bouba_kiki_value,
    sound_dict,
    num_sounds=5
):
    sorted_sounds = sorted(
        sound_dict.items(),
        key=lambda x: abs(x[1] - bouba_kiki_value)
    )
    return [sound[0] for sound in sorted_sounds[:num_sounds]]


def generate_pseudoword(
    num_syllables,
    consonants,
    vowels
):
    pseudoword = ""
    for _ in range(num_syllables):
        pseudoword += random.choice(consonants) + random.choice(vowels)
    return pseudoword


def evaluate_pseudoword(
    pseudoword,
    bouba_kiki_value,
    sound_dict
):
    total_bouba_kiki_value = 0
    for char in pseudoword:
        if char in sound_dict["vowels"]:
            total_bouba_kiki_value += sound_dict["vowels"][char]
        else:
            total_bouba_kiki_value += sound_dict["consonants"][char]
    avg_bouba_kiki_value = total_bouba_kiki_value / len(pseudoword)

    error = abs(avg_bouba_kiki_value - bouba_kiki_value)
    return avg_bouba_kiki_value, error


# Main function
def pseudoword_generator(
    bouba_kiki_value,
    num_polylines,
    filename='sound_mappings.json'
):
    sound_dict = load_sound_mappings(filename)
    num_syllables = get_syllable_count(num_polylines)
    selected_consonants = select_sounds(
        bouba_kiki_value,
        sound_dict["consonants"]
    )
    selected_vowels = select_sounds(
        bouba_kiki_value,
        sound_dict["vowels"],
        num_sounds=3
    )

    return generate_pseudoword(
        num_syllables,
        selected_consonants,
        selected_vowels
    )


if __name__ == "__main__":
    filename = 'utils/sound_mappings.json'
    num_tests = 10
    num_polylines = [random.randint(1, 1000) for _ in range(num_tests)]
    bouba_kiki_value = [random.random() for _ in range(num_tests)]
    sound_dict = load_sound_mappings(filename)
    scores = []
    accumulated_error = 0

    for i in range(num_tests):
        pseudoword = pseudoword_generator(
            bouba_kiki_value[i],
            num_polylines[i],
            filename
        )
        score, error = evaluate_pseudoword(
            pseudoword,
            bouba_kiki_value[i],
            sound_dict
        )
        scores.append(score)
        print(f"Lines: {num_polylines[i]:>4}", end=" | ")
        print(f"Bouba-Kiki Value: {bouba_kiki_value[i]:.4f}", end=" | ")
        print(f"Pseudoword: {pseudoword:<8}", end=" | ")
        print(f"Score: {score:.4f}", end=" | ")
        print(f"Error: {error:.4f}")
        accumulated_error += error

    print(f"\nAverage error: {accumulated_error / num_tests:.4f}")
