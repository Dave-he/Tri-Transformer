import { http, HttpResponse } from 'msw';

export const authHandlers = [
  http.post('http://localhost:8002/api/v1/auth/login', async ({ request }) => {
    const body = await request.json() as { username: string; password: string };
    if (body.username === 'wrong' || body.password === 'wrong') {
      return HttpResponse.json({ message: 'Invalid credentials' }, { status: 401 });
    }
    return HttpResponse.json({
      token: 'mock-token-abc123',
      user: { id: 'user-1', username: body.username, email: `${body.username}@test.com` },
    });
  }),

  http.post('http://localhost:8002/api/v1/auth/register', async ({ request }) => {
    const body = await request.json() as { username: string; password: string; email: string };
    return HttpResponse.json({
      user: { id: 'user-2', username: body.username, email: body.email },
    }, { status: 201 });
  }),

  http.post('http://localhost:8002/api/v1/auth/logout', () => {
    return HttpResponse.json({ message: 'Logged out' });
  }),
];
