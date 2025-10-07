import { useState } from 'react';
import { LogIn, UserPlus, Loader2, Sparkles, ChefHat, UtensilsCrossed } from 'lucide-react';

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
    <div className="min-h-screen relative overflow-hidden">
      {/* Background Image with Overlay */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: "url('https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=1920&h=1080&fit=crop')",
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-orange-900/95 via-red-900/90 to-yellow-900/95"></div>
      </div>

      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-orange-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-yellow-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-red-500/10 rounded-full blur-3xl animate-pulse delay-500"></div>
      </div>

      {/* Content Container */}
      <div className="relative min-h-screen flex items-center justify-center p-4">
        <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
          {/* Left Side - Branding */}
          <div className="hidden lg:block text-white space-y-8 animate-fade-in">
            <div className="flex items-center gap-3 mb-8">
              <div className="bg-white/20 backdrop-blur-md p-4 rounded-2xl">
                <ChefHat className="w-12 h-12 text-white" />
              </div>
              <div>
                <h1 className="text-4xl font-bold">Restaurant ML</h1>
                <p className="text-orange-200">Analysis Platform</p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="flex items-start gap-4 bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/20 hover:bg-white/20 transition-all duration-300">
                <div className="bg-orange-500/30 p-3 rounded-xl">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">AI-Powered Predictions</h3>
                  <p className="text-orange-100 text-sm">
                    Leverage machine learning to predict ratings, classify cuisines, and get personalized recommendations
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4 bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/20 hover:bg-white/20 transition-all duration-300">
                <div className="bg-red-500/30 p-3 rounded-xl">
                  <UtensilsCrossed className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">Comprehensive Analytics</h3>
                  <p className="text-orange-100 text-sm">
                    Explore location-based insights, geospatial visualizations, and deep restaurant analytics
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4 bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/20 hover:bg-white/20 transition-all duration-300">
                <div className="bg-yellow-500/30 p-3 rounded-xl">
                  <LogIn className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-bold text-lg mb-2">Secure Access</h3>
                  <p className="text-orange-100 text-sm">
                    Sign in to access all features and save your preferences for a personalized experience
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-8 border-t border-white/20">
              <p className="text-orange-200 text-sm">
                🎓 Part of the <span className="font-bold text-white">Cognifyz Technologies</span> Machine Learning Internship Program
              </p>
            </div>
          </div>

          {/* Right Side - Auth Form */}
          <div className="w-full">
            <div className="bg-white/95 backdrop-blur-xl shadow-2xl rounded-3xl p-8 md:p-10 border border-orange-200 animate-fade-in-up">
              {/* Mobile Logo */}
              <div className="lg:hidden flex items-center justify-center mb-8">
                <div className="bg-gradient-to-br from-orange-500 to-red-500 p-4 rounded-2xl">
                  {isLogin ? (
                    <LogIn className="w-10 h-10 text-white" />
                  ) : (
                    <UserPlus className="w-10 h-10 text-white" />
                  )}
                </div>
              </div>

              {/* Desktop Icon */}
              <div className="hidden lg:flex items-center justify-center mb-8">
                <div className="bg-gradient-to-br from-orange-500 to-red-500 p-5 rounded-2xl shadow-lg">
                  {isLogin ? (
                    <LogIn className="w-12 h-12 text-white" />
                  ) : (
                    <UserPlus className="w-12 h-12 text-white" />
                  )}
                </div>
              </div>
              
              <h2 className="text-3xl md:text-4xl font-bold text-center bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent mb-3">
                {isLogin ? 'Welcome Back' : 'Create Account'}
              </h2>
              <p className="text-center text-gray-600 mb-8 text-base">
                {isLogin ? 'Sign in to continue your journey' : 'Join us to explore AI-powered insights'}
              </p>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label htmlFor="username" className="block text-sm font-semibold text-gray-700 mb-2">
                    Username
                  </label>
                  <input
                    type="text"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-orange-100 focus:border-orange-500 transition-all duration-200 text-gray-900"
                    placeholder="Enter your username"
                    required
                  />
                </div>

                <div>
                  <label htmlFor="password" className="block text-sm font-semibold text-gray-700 mb-2">
                    Password
                  </label>
                  <input
                    type="password"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-orange-100 focus:border-orange-500 transition-all duration-200 text-gray-900"
                    placeholder="Enter your password"
                    required
                    minLength={6}
                  />
                  {!isLogin && (
                    <p className="text-xs text-gray-500 mt-2">
                      Password must be at least 6 characters long
                    </p>
                  )}
                </div>

                {error && (
                  <div className="p-4 bg-red-50 border-2 border-red-200 rounded-xl animate-shake">
                    <p className="text-red-600 text-sm font-medium">{error}</p>
                  </div>
                )}

                <button
                  type="submit"
                  className={`w-full flex justify-center items-center py-4 px-6 rounded-xl text-white font-semibold shadow-lg transition-all duration-300 transform hover:-translate-y-0.5 ${
                    loading
                      ? 'bg-gradient-to-r from-orange-400 to-red-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 hover:shadow-xl'
                  }`}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>{isLogin ? 'Sign In' : 'Create Account'}</>
                  )}
                </button>
              </form>

              <div className="mt-8 text-center">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-300"></div>
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-4 bg-white text-gray-500">or</span>
                  </div>
                </div>

                <button
                  onClick={() => {
                    setIsLogin(!isLogin);
                    setError(null);
                    setFormData({ username: '', password: '' });
                  }}
                  className="mt-6 text-orange-600 hover:text-orange-700 font-semibold text-base transition-colors duration-200"
                >
                  {isLogin
                    ? "Don't have an account? Sign up"
                    : 'Already have an account? Sign in'}
                </button>
              </div>
            </div>

            {/* Mobile Info Cards */}
            <div className="lg:hidden mt-6 space-y-4">
              <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl border border-white/20 text-white">
                <p className="text-sm text-center">
                  🎓 Part of the <span className="font-bold">Cognifyz Technologies</span> ML Internship
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes fade-in-up {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }
        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }
        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out;
        }
        .animate-shake {
          animation: shake 0.3s ease-in-out;
        }
        .delay-500 {
          animation-delay: 500ms;
        }
        .delay-1000 {
          animation-delay: 1000ms;
        }
      `}</style>
    </div>
  );
}

export default AuthPage;