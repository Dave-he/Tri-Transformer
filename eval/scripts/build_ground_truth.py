import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.ground_truth.document_qa_generator import DocumentQAGenerator
from eval.ground_truth.dual_model_validator import DualModelValidator
from eval.ground_truth.kg_triple_extractor import KGTripleExtractor
from eval.ground_truth.consistency_checker import ConsistencyChecker
from eval.ground_truth.fusion_engine import GTFusionEngine
from eval.ground_truth.gt_versioning import GTVersioning
from eval.ground_truth.schema import SourceType


def load_documents_from_dir(docs_dir: str) -> list:
    documents = []
    doc_id = 0
    for root, dirs, files in os.walk(docs_dir):
        for fname in files:
            if fname.endswith((".txt", ".md")):
                fpath = os.path.join(root, fname)
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    documents.append({"id": f"doc_{doc_id}", "content": content})
                    doc_id += 1
    return documents


def main():
    parser = argparse.ArgumentParser(description="Build Ground Truth dataset")
    parser.add_argument("--docs_dir", required=True, help="Directory containing documents")
    parser.add_argument("--output_dir", default="eval/data/gt_versions", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=-1, help="Max number of samples (-1 for all)")
    args = parser.parse_args()

    print(f"Loading documents from {args.docs_dir}")
    documents = load_documents_from_dir(args.docs_dir)
    print(f"Loaded {len(documents)} documents")

    generator = DocumentQAGenerator(min_qa_per_1000_chars=5)
    qa_pairs = generator.generate(documents)
    print(f"Generated {len(qa_pairs)} QA pairs")

    extractor = KGTripleExtractor()
    for doc in documents:
        triples = extractor.extract(doc["content"])
        for triple in triples:
            qa = extractor.triple_to_qa(triple)
            qa["source_doc_id"] = doc["id"]
            qa["source_type"] = SourceType.KG_TRIPLE
            qa_pairs.append(qa)

    for qa in qa_pairs:
        if "source_type" not in qa:
            qa["source_type"] = SourceType.DOCUMENT_QA

    fusion = GTFusionEngine()
    fused = fusion.fuse(qa_pairs)

    raw_for_storage = [item.model_dump() for item in fused]
    if args.num_samples > 0:
        raw_for_storage = raw_for_storage[:args.num_samples]

    versioning = GTVersioning(storage_dir=args.output_dir)
    version_id = versioning.save(raw_for_storage)
    print(f"Saved Ground Truth dataset: version={version_id}, samples={len(raw_for_storage)}")

    export_path = os.path.join(args.output_dir, f"gt_latest.jsonl")
    versioning.export_jsonlines(version_id, export_path)
    print(f"Exported to {export_path}")


if __name__ == "__main__":
    main()
