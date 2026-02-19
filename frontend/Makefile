.PHONY: dev build test lint typecheck check generate-api clean install

dev:
	pnpm dev

build:
	pnpm build

test:
	pnpm vitest run

lint:
	pnpm biome check src/

typecheck:
	pnpm tsc --noEmit

check: lint typecheck

generate-api:
	pnpm orval

clean:
	rm -rf dist/ node_modules/.vite/

install:
	pnpm install
