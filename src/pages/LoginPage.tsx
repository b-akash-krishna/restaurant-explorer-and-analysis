import { useState } from 'react';
import { LogIn, UserPlus, Loader2 } from 'lucide-react';

interface AuthPageProps {
  onAuthSuccess: () => void;
}

function AuthPage({ onAuthSuccess }: AuthPageProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (isLogin) {
        // Login
        const formBody = new URLSearchParams();
        formBody.append('username', formData.username);
        formBody.append('password', formData.password);

        const response = await fetch('http://localhost:8000/api/auth/token', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formBody.toString(),
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || 'Login failed');
        }

        const data = await response.json();
        localStorage.setItem('access_token', data.access_token);
        onAuthSuccess();
      } else {
        // Register
        const response = await fetch('http://localhost:8000/api/auth/register', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            username: formData.username,
            password: formData.password,
            role: 'user',
          }),
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || 'Registration failed');
        }

        // Auto-login after registration
        const formBody = new URLSearchParams();
        formBody.append('username', formData.username);
        formBody.append('password', formData.password);

        const loginResponse = await fetch('http://localhost:8000/api/auth/token', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formBody.toString(),
        });

        if (loginResponse.ok) {
          const loginData = await loginResponse.json();
          localStorage.setItem('access_token', loginData.access_token);
          onAuthSuccess();
        }
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-orange-100 flex items-center justify-center p-4">
      <div className="bg-white shadow-2xl rounded-2xl p-8 w-full max-w-md">
        <div className="flex items-center justify-center mb-6">
          {isLogin ? (
            <LogIn className="w-12 h-12 text-orange-500" />
          ) : (
            <UserPlus className="w-12 h-12 text-orange-500" />
          )}
        </div>
        
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-2">
          {isLogin ? 'Welcome Back' : 'Create Account'}
        </h2>
        <p className="text-center text-gray-500 mb-8">
          {isLogin ? 'Sign in to continue' : 'Sign up to get started'}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-1">
              Username
            </label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              required
              minLength={6}
            />
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          )}

          <button
            type="submit"
            className={`w-full flex justify-center items-center py-3 px-4 rounded-lg text-white font-medium ${
              loading
                ? 'bg-orange-400 cursor-not-allowed'
                : 'bg-orange-600 hover:bg-orange-700'
            }`}
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Processing...
              </>
            ) : (
              <>{isLogin ? 'Sign In' : 'Sign Up'}</>
            )}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            onClick={() => {
              setIsLogin(!isLogin);
              setError(null);
            }}
            className="text-orange-600 hover:text-orange-700 text-sm font-medium"
          >
            {isLogin
              ? "Don't have an account? Sign up"
              : 'Already have an account? Sign in'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default AuthPage;