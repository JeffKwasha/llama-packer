/**
 * llama-swap Plugin for OpenCode
 *
 * Queries llama-swap (LLAMASWAP_URL) instance, filters models for tool calling
 * and injects them into opencode's 'llama-swap' (LLAMASWAP_PROVIDER). 
 * Also provides tools (list, status, unload) 
 *
 * MIT License
 * Copyright (c) 2025,2026 Cory Bryan, Jeffrey Kwasha
 *
 * 2026-09 - overhaul to filter non-agentic models, shrink long names, fix context, capabilities
 */

import type { Plugin, Config, PluginInput } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"

const FETCH_TIMEOUT = 300
const DEFAULT_OUTPUT_LIMIT = 65536
const DEFAULT_CONTEXT_LIMIT = 131072
const DEFAULT_URL = "http://localhost:8080"
const DEFAULT_PROVIDER = "llama-swap"
const PROVIDER_KEYS = ["llamaSwap", "llama.cpp"]

function stripBaseUrl(u: string): string {
  return u.replace(/\/v1\/?$/, "").replace(/\/+$/, "")
}

async function resolveTarget(client: PluginInput["client"], config?: Config): Promise<{ url: string; provider: string }> {
  let provider = process.env.LLAMASWAP_PROVIDER || DEFAULT_PROVIDER
  for (const key of PROVIDER_KEYS) {
    if (config?.provider?.[key]) { provider = key; break }
  }

  let url = ""
  const fromConfig = PROVIDER_KEYS.reduce((acc, k) => acc ?? config?.provider?.[k]?.options?.baseURL, undefined as string | undefined)
    ?? config?.provider?.[DEFAULT_PROVIDER]?.options?.baseURL
  if (fromConfig) url = stripBaseUrl(fromConfig)
  else if (process.env.LLAMASWAP_URL) url = stripBaseUrl(process.env.LLAMASWAP_URL)
  else {
    try {
      const { data: live } = await client.config.get().catch(() => ({}))
      if (live) {
        for (const key of [...PROVIDER_KEYS, DEFAULT_PROVIDER]) {
          const b = (live as any).provider?.[key]?.options?.baseURL
          if (b) { url = stripBaseUrl(b); break }
        }
      }
    } catch {}
    if (!url) url = DEFAULT_URL
  }
  return { url, provider }
}

async function fetchJSON<T>(url: string, path: string, opts?: RequestInit): Promise<T> {
  const resp = await fetch(`${url}${path}`, { ...opts, signal: AbortSignal.timeout(FETCH_TIMEOUT) })
  if (!resp.ok) {
    const body = await resp.text().catch(() => "")
    throw new Error(`HTTP ${resp.status} ${resp.statusText}${body ? `: ${body.slice(0, 500)}` : ""}`)
  }
  return resp.json() as Promise<T>
}

interface LlamaSwapModel {
  id: string
  name?: string
  context_length?: number
  context_window?: number
  architecture?: { input_modalities?: string[]; output_modalities?: string[] }
  capabilities?: { vision?: boolean; function_calling?: boolean; tools?: boolean; reranker?: boolean; [k: string]: unknown }
  supported_parameters?: string[]
  meta?: {
    n_ctx?: number
    llamaswap?: {
      ctx_size?: number | string
      freethought?: number
      modes?: string[]
      quantization?: string
      parameters?: string
      [k: string]: unknown
    }
  }
}

function resolveContext(m: LlamaSwapModel): number {
  const n = m.meta?.n_ctx; if (typeof n === "number" && n > 0) return n
  if (typeof m.context_length === "number" && m.context_length > 0) return m.context_length
  if (typeof m.context_window === "number" && m.context_window > 0) return m.context_window
  const cs = m.meta?.llamaswap?.ctx_size; if (typeof cs === "number" && cs > 0) return cs
  if (typeof cs === "string") { const v = parseInt(cs, 10); if (v > 0) return v }
  return DEFAULT_CONTEXT_LIMIT
}

function isChatModel(m: LlamaSwapModel): boolean {
  const caps = m.capabilities || {}
  const sp = m.supported_parameters || []
  return !!caps.function_calling || !!caps.tools || sp.includes("tools")
}

function hasVision(m: LlamaSwapModel): boolean {
  return !!m.capabilities?.vision || !!m.architecture?.input_modalities?.includes("image") || !!m.meta?.llamaswap?.multimodal
}

function hasThinking(m: LlamaSwapModel): boolean {
  const x = m.meta?.llamaswap
  return !!x?.thinking || typeof x?.freethought === "number" || !!x?.modes?.includes("thinking")
}

function displayNameForId(id: string): string {
  const i = id.indexOf(":"); const base = i >= 0 ? id.slice(0, i) : id; const alias = i >= 0 ? id.slice(i + 1) : null
  return alias ? `${base} (${alias})` : base
}

function log(client: PluginInput["client"], level: string, message: string, extra?: Record<string, unknown>) {
  client.app.log({ body: { service: "llamaswap", level, message, extra } }).catch(() => {})
}

export const LlamaSwapPlugin: Plugin = async ({ client: _client }: PluginInput) => {
  const client = _client
  return {
    config: async (cfg: Config) => {
      try {
        const { url, provider } = await resolveTarget(client, cfg)
        const data = await fetchJSON<{ data: LlamaSwapModel[] }>(url, "/v1/models")
        if (!data || !Array.isArray((data as any).data)) { log(client, "warn", `unexpected shape from ${url}/v1/models`); return }
        if (!cfg.provider) cfg.provider = {}
        if (!cfg.provider[provider]) cfg.provider[provider] = { npm: "@ai-sdk/openai-compatible", name: "llama-swap", options: { baseURL: `${url}/v1` } } as any
        else if (!(cfg.provider[provider] as any).options?.baseURL) (cfg.provider[provider] as any).options = { ...((cfg.provider[provider] as any).options || {}), baseURL: `${url}/v1` }
        if (!cfg.provider[provider].models) cfg.provider[provider].models = {}
        const existing = cfg.provider[provider].models!
        for (const m of data.data) {
          if (!m || typeof m.id !== "string" || !isChatModel(m) || existing[m.id]) continue
          const ctx = resolveContext(m); const out = Math.max(4096, Math.min(Math.floor(ctx / 2), DEFAULT_OUTPUT_LIMIT))
          const modelCfg: Record<string, unknown> = { name: displayNameForId(m.id), limit: { context: ctx, output: out } }
          if (hasVision(m)) modelCfg.attachment = true
          if (hasThinking(m)) { modelCfg.reasoning = true; modelCfg.variants = { off: { reasoning: false } } as any }
          existing[m.id] = modelCfg as any
        }
      } catch (err) {
        const { url } = await resolveTarget(client).catch(() => ({ url: DEFAULT_URL }))
        log(client, "warn", `skipping discovery — failed to fetch from ${url}: ${err}`)
      }
    },

    tool: {
      llamaswap_models: tool({
        description: "List all available models on the llama-swap server with metadata (size, quantization, context, backend)",
        args: tool.schema.object({}),
        async execute() {
          try {
            const { url } = await resolveTarget(client)
            const data = await fetchJSON<{ data: LlamaSwapModel[] }>(url, "/v1/models")
            if (!data || !Array.isArray((data as any).data)) return `Failed to list models: unexpected shape from ${url}/v1/models`
            const lines: string[] = []
            for (const m of data.data) {
              if (!m || typeof m.id !== "string" || !isChatModel(m)) continue
              const ctx = resolveContext(m); const parts = [m.id]
              const disp = displayNameForId(m.id); if (disp !== m.id) parts.push(`"${disp}"`)
              const q = (m.meta?.llamaswap as any)?.quantization; const p = (m.meta?.llamaswap as any)?.parameters
              if (p) parts.push(`size=${p}`); if (q) parts.push(`quant=${q}`)
              parts.push(`ctx=${ctx}`)
              const caps: string[] = []; if (hasVision(m)) caps.push("vision"); if (hasThinking(m)) caps.push("thinking"); if (m.capabilities?.function_calling || m.capabilities?.tools || (m.supported_parameters || []).includes("tools")) caps.push("tool_calling")
              if (caps.length) parts.push(`caps=[${caps.join(",")}]`)
              lines.push(parts.join(" | "))
            }
            return `${lines.length} chat models available (filtered from ${data.data.length} total):\n\n${lines.join("\n")}`
          } catch (err) {
            const { url } = await resolveTarget(client).catch(() => ({ url: DEFAULT_URL }))
            return `Failed to list models from ${url}: ${err} — is llama-swap running?`
          }
        },
      }),
      llamaswap_status: tool({
        description: "Show currently running/loaded models on the llama-swap server",
        args: tool.schema.object({}),
        async execute() {
          try {
            const { url } = await resolveTarget(client)
            const data = await fetchJSON<{ running: unknown[] }>(url, "/running")
            if (!data || !Array.isArray((data as any).running) || data.running.length === 0) return "No models currently loaded."
            return `Running models:\n${JSON.stringify(data.running, null, 2)}`
          } catch (err) {
            const { url } = await resolveTarget(client).catch(() => ({ url: DEFAULT_URL }))
            return `Failed to get status from ${url}: ${err} — is llama-swap running?`
          }
        },
      }),
      llamaswap_unload: tool({
        description: "Unload the current model from the llama-swap server to free VRAM",
        args: tool.schema.object({}),
        async execute() {
          try {
            const { url } = await resolveTarget(client)
            const resp = await fetch(`${url}/api/models/unload`, { method: "POST", signal: AbortSignal.timeout(FETCH_TIMEOUT) })
            if (resp.ok) return "Model unloaded successfully."
            const body = await resp.text().catch(() => "")
            return `Unload failed: HTTP ${resp.status} ${resp.statusText}${body ? `: ${body.slice(0, 500)}` : ""}`
          } catch (err) {
            const { url } = await resolveTarget(client).catch(() => ({ url: DEFAULT_URL }))
            return `Failed to unload model at ${url}: ${err} — is llama-swap running?`
          }
        },
      }),
    },
  }
}
