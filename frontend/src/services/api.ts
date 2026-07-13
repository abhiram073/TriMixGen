import axios from 'axios';

const API_BASE_URL = '/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface GenerateRequest {
  prompt: string;
  style?: string;
  english_usage?: string;
  temperature?: number;
}

export interface GenerateResponse {
  generated_text: string;
  rendered_prompt: string;
  language_tags: string[];
  cmi: number;
  latency: number;
  inference_config: Record<string, any>;
  token_count: number;
}

export interface TagRequest {
  text: string;
}

export interface TagResponse {
  tokens: string[];
  labels: string[];
  confidence: number;
  cmi: number;
}

export interface HealthResponse {
  status: string;
  generator_loaded: boolean;
  indicbert_loaded: boolean;
  uptime: number;
  device: string;
}

export interface ModelInfoResponse {
  model_version: string;
  tokenizer: string;
  parameter_count: number;
  generation_configuration: Record<string, any>;
  deployment_metadata: Record<string, any>;
}

export const trimixgenAPI = {
  generate: async (request: GenerateRequest): Promise<GenerateResponse> => {
    const { data } = await api.post<GenerateResponse>('/generate', request);
    return data;
  },
  
  tagText: async (request: TagRequest): Promise<TagResponse> => {
    const { data } = await api.post<TagResponse>('/tag', request);
    return data;
  },
  
  getHealth: async (): Promise<HealthResponse> => {
    const { data } = await api.get<HealthResponse>('/health');
    return data;
  },
  
  getModelInfo: async (): Promise<ModelInfoResponse> => {
    const { data } = await api.get<ModelInfoResponse>('/model-info');
    return data;
  }
};
