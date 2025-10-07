import { TrendingUp, ThumbsUp, Utensils, MapPin, ArrowRight, Sparkles } from 'lucide-react';

type Page = 'home' | 'rating' | 'recommendations' | 'cuisine' | 'location';

interface HomePageProps {
  onNavigate: (page: Page) => void;
}

function HomePage({ onNavigate }: HomePageProps) {
  const tasks = [
    {
      id: 'rating' as Page,
      title: 'Restaurant Rating Prediction',
      description: 'Predict restaurant ratings using machine learning regression models based on features like votes, cost, and amenities.',
      icon: TrendingUp,
      color: 'bg-blue-500',
      gradient: 'from-blue-500 to-blue-600',
      image: 'https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=800&h=600&fit=crop',
    },
    {
      id: 'recommendations' as Page,
      title: 'Restaurant Recommendations',
      description: 'Get personalized restaurant recommendations based on cuisine preferences, location, and price range using content-based filtering.',
      icon: ThumbsUp,
      color: 'bg-green-500',
      gradient: 'from-green-500 to-green-600',
      image: 'https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=800&h=600&fit=crop',
    },
    {
      id: 'cuisine' as Page,
      title: 'Cuisine Classification',
      description: 'Classify restaurant cuisines using multi-class classification models trained on restaurant features and characteristics.',
      icon: Utensils,
      color: 'bg-orange-500',
      gradient: 'from-orange-500 to-orange-600',
      image: 'https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=800&h=600&fit=crop',
    },
    {
      id: 'location' as Page,
      title: 'Location-based Analysis',
      description: 'Explore geospatial visualizations and analyze restaurant distributions, ratings, and trends across different cities and localities.',
      icon: MapPin,
      color: 'bg-red-500',
      gradient: 'from-red-500 to-red-600',
      image: 'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=800&h=600&fit=crop',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      {/* Hero Section */}
      <div 
        className="relative bg-cover bg-center bg-no-repeat min-h-screen flex items-center"
        style={{
          backgroundImage: "url('https://images.unsplash.com/photo-1552566626-52f8b828add9?w=1920&h=1080&fit=crop')",
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-black/80 via-black/70 to-black/60"></div>
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32">
          <div className="flex items-center gap-2 mb-6 animate-fade-in">
            <Sparkles className="w-8 h-8 text-amber-400" />
            <span className="text-amber-400 font-semibold tracking-wide uppercase text-sm">AI-Powered Insights</span>
          </div>
          
          <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold text-white mb-8 leading-tight animate-fade-in-up">
            Restaurant ML
            <br />
            <span className="bg-gradient-to-r from-amber-400 via-orange-500 to-red-500 bg-clip-text text-transparent">
              Analysis Platform
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-200 max-w-3xl mb-12 leading-relaxed animate-fade-in-up-delay">
            Comprehensive machine learning solutions for restaurant data analysis. Explore rating predictions,
            personalized recommendations, cuisine classification, and location-based insights powered by advanced AI.
          </p>
          
          <div className="flex flex-wrap gap-4 animate-fade-in-up-delay-2">
            <button 
              onClick={() => onNavigate('rating')}
              className="px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-600 text-white font-semibold rounded-full hover:shadow-2xl hover:scale-105 transition-all duration-300 flex items-center gap-2"
            >
              Get Started
              <ArrowRight className="w-5 h-5" />
            </button>
            <button className="px-8 py-4 bg-white/10 backdrop-blur-md text-white font-semibold rounded-full border-2 border-white/30 hover:bg-white/20 transition-all duration-300">
              Learn More
            </button>
          </div>
        </div>
        
        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/70 rounded-full mt-2 animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold text-gray-900 mb-4">
            Explore Our <span className="bg-gradient-to-r from-orange-500 to-red-500 bg-clip-text text-transparent">Features</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Four powerful machine learning models designed to revolutionize restaurant analytics
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {tasks.map((task, index) => {
            const Icon = task.icon;
            return (
              <div
                key={task.id}
                className="group relative overflow-hidden rounded-2xl cursor-pointer transform hover:-translate-y-2 transition-all duration-500 hover:shadow-2xl"
                onClick={() => onNavigate(task.id)}
                style={{
                  animationDelay: `${index * 100}ms`,
                }}
              >
                {/* Background Image */}
                <div 
                  className="absolute inset-0 bg-cover bg-center transition-transform duration-700 group-hover:scale-110"
                  style={{ backgroundImage: `url('${task.image}')` }}
                ></div>
                
                {/* Gradient Overlay */}
                <div className={`absolute inset-0 bg-gradient-to-br ${task.gradient} opacity-90 group-hover:opacity-95 transition-opacity duration-300`}></div>
                
                {/* Content */}
                <div className="relative p-8 min-h-[320px] flex flex-col justify-between">
                  <div>
                    <div className="flex items-start justify-between mb-6">
                      <div className="p-4 bg-white/20 backdrop-blur-md rounded-xl group-hover:bg-white/30 transition-all duration-300">
                        <Icon className="h-8 w-8 text-white" />
                      </div>
                      <ArrowRight className="h-6 w-6 text-white/70 group-hover:text-white group-hover:translate-x-2 transition-all duration-300" />
                    </div>
                    
                    <h3 className="text-2xl font-bold text-white mb-4 group-hover:translate-x-1 transition-transform duration-300">
                      {task.title}
                    </h3>
                    
                    <p className="text-white/90 text-base leading-relaxed group-hover:text-white transition-colors duration-300">
                      {task.description}
                    </p>
                  </div>
                  
                  <div className="mt-6 flex items-center text-white/80 group-hover:text-white transition-colors duration-300">
                    <span className="text-sm font-semibold">Explore Feature</span>
                    <ArrowRight className="w-4 h-4 ml-2 group-hover:ml-3 transition-all duration-300" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* About Section */}
      <div className="relative py-24 overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-fixed"
          style={{
            backgroundImage: "url('https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?w=1920&h=1080&fit=crop')",
          }}
        ></div>
        <div className="absolute inset-0 bg-gradient-to-b from-black/85 to-black/90"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                About This Project
              </h2>
              <div className="w-24 h-1 bg-gradient-to-r from-amber-400 to-orange-500 mx-auto"></div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-8 md:p-12 border border-white/20 shadow-2xl">
              <p className="text-gray-200 text-lg mb-8 leading-relaxed">
                This platform is part of the <span className="text-amber-400 font-semibold">Cognifyz Technologies Machine Learning Internship Program</span>.
                It demonstrates practical applications of various ML techniques on real-world restaurant data, combining cutting-edge technology with intuitive design.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <div className="w-2 h-2 bg-amber-400 rounded-full"></div>
                    Technologies Used
                  </h3>
                  <ul className="text-gray-300 space-y-3">
                    <li className="flex items-center gap-2">
                      <span className="text-amber-400">→</span>
                      <span>Backend: Python, FastAPI</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-amber-400">→</span>
                      <span>Frontend: React, TypeScript, Tailwind CSS</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-amber-400">→</span>
                      <span>ML: Scikit-learn, Pandas, NumPy</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-amber-400">→</span>
                      <span>Deployment: Docker, GitHub Actions</span>
                    </li>
                  </ul>
                </div>
                
                <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                    ML Techniques
                  </h3>
                  <ul className="text-gray-300 space-y-3">
                    <li className="flex items-center gap-2">
                      <span className="text-orange-400">→</span>
                      <span>Regression Analysis</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-orange-400">→</span>
                      <span>Content-based Filtering</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-orange-400">→</span>
                      <span>Multi-class Classification</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-orange-400">→</span>
                      <span>Geospatial Analysis</span>
                    </li>
                  </ul>
                </div>
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
        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }
        .animate-fade-in-up {
          animation: fade-in-up 1s ease-out 0.2s backwards;
        }
        .animate-fade-in-up-delay {
          animation: fade-in-up 1s ease-out 0.4s backwards;
        }
        .animate-fade-in-up-delay-2 {
          animation: fade-in-up 1s ease-out 0.6s backwards;
        }
      `}</style>
    </div>
  );
}

export default HomePage;