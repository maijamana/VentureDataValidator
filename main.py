import json
from text_preprocessor import TextPreprocessor
from semantic_segmenter import SemanticSegmenter
from faiss_utils import build_faiss_index, search_all_above_threshold


preprocessor = TextPreprocessor()
segmenter = SemanticSegmenter()


def process_citations(citations, source_text):
    res = []
    segments = segmenter.segment(source_text, 0.75)
    for i in range(len(segments)):
        segments[i] = preprocessor.preprocess_document(segments[i])

    for quote in citations:
        quote = preprocessor.preprocess_document(quote)

        index, embeddings, all_segments, model = build_faiss_index(segments)
        results = search_all_above_threshold(quote, index, all_segments, model, threshold=0.7)

        if results:
            res.append('likely true')
        else:
            res.append('likely untrue')

    return res


def main():
    print('==========START PROCESSING==========')
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for obj in data:
        citations = []
        location_map = []

        for key, value in obj.items():
            if key == "source_text":
                continue
            if isinstance(value, list):
                for i, entry in enumerate(value):
                    if "citation" in entry:
                        citations.append(entry["citation"])
                        location_map.append((key, i)) 
            elif isinstance(value, dict):
                if "citation" in value:
                    citations.append(value["citation"])
                    location_map.append((key, None))

        source_text = obj.get("source_text", "")

        probabilities = process_citations(citations, source_text)

        for (key, idx), prob in zip(location_map, probabilities):
            if idx is not None:
                obj[key][idx]["probability_of_veracity"] = prob
            else:
                obj[key]["probability_of_veracity"] = prob

    with open("data_updated.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
