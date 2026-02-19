.PHONY: dev build test e2e lint typecheck check generate-api clean install

dev:
	pnpm dev

build:
	pnpm build

test:
	pnpm vitest run

e2e:
	pnpm playwright test

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
