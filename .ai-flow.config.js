export default {
  preset: 'balanced',
  platform: 'codeflicker',
  project: {
    type: 'fullstack',
    name: 'Tri-Transformer',
    stack: {
      frontend: 'react',
      backend: 'fastapi',
    },
  },
  directories: {
    snippets: '.codeflicker/snippets',
    docs: 'docs',
    tasks: 'docs/tasks',
    research: 'docs/research',
  },
  commands: {
    'frontend:dev': 'cd frontend && pnpm dev',
    'frontend:build': 'cd frontend && pnpm build',
    'frontend:test': 'cd frontend && pnpm test',
    'frontend:lint': 'cd frontend && pnpm lint',
    'frontend:typecheck': 'cd frontend && pnpm typecheck',
    'backend:dev': 'cd backend && uvicorn app.main:app --reload',
    'backend:test': 'cd backend && pytest',
    'docker:up': 'docker-compose up -d',
    'docker:down': 'docker-compose down',
  },
}
