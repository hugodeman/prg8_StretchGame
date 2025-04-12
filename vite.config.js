import { defineConfig } from "vite";

/** @type {import('vite').UserConfig} */

export default defineConfig({
    base: "./",

    build: {
        outDir: '/docs',
        emptyOutDir: true, // empty the build dir before new build
    }
});