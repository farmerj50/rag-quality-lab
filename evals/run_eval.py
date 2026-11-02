"""Live evaluation of configured retrieval and answerer. Run after ingesting sample docs."""
import json
from pathlib import Path
from app.service import RagService


def main():
    svc = RagService()
    records = [json.loads(line) for line in Path('evals/golden.jsonl').read_text().splitlines()]
    hits = 0
    abstains = 0
    for item in records:
        response = svc.ask(item['question'])
        sources = response['sources']
        retrieved = any(s['source'] == item['expected_source'] for s in sources) if item['answerable'] else not sources
        hits += int(retrieved)
        if not item['answerable']:
            abstains += int(not sources)
        print(json.dumps({'question': item['question'], 'retrieval_pass': retrieved,
                          'answer': response['answer'], 'terms_present': all(
                              term.lower() in response['answer'].lower() for term in item['expected_terms'])}))
    print(json.dumps({'retrieval_checks_passed': hits, 'total': len(records),
                      'unanswerable_queries_abstained': abstains}))
    print('Note: keyword checks are not a sufficient hallucination/groundedness metric.')


if __name__ == '__main__':
    main()
