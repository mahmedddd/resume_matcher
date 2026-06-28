import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const internshipApi = {
  getInternships: (params: any) => api.get('/internships', { params }),
  getStats: () => api.get('/stats'),
  uploadCV: (formData: FormData) => api.post('/upload-cv', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  triggerScrape: (sources?: string[]) => api.post('/scrape', { sources }),
  getScrapeStatus: () => api.get('/scrape/status'),
};
