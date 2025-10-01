import { useState } from 'react';
import { Menu, X, Home, TrendingUp, ThumbsUp, Utensils, MapPin } from 'lucide-react';
import RatingPrediction from './pages/RatingPrediction';
import Recommendations from './pages/Recommendations';
import CuisineClassification from './pages/CuisineClassification';
import LocationAnalysis from './pages/LocationAnalysis';
import HomePage from './pages/HomePage';

type Page = 'home' | 'rating' | 'recommendations' | 'cuisine' | 'location';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');
  const [menuOpen, setMenuOpen] = useState(false);

  const navigation = [
    { id: 'home' as Page, name: 'Home', icon: Home },
    { id: 'rating' as Page, name: 'Rating Prediction', icon: TrendingUp },
    { id: 'recommendations' as Page, name: 'Recommendations', icon: ThumbsUp },
    { id: 'cuisine' as Page, name: 'Cuisine Classification', icon: Utensils },
    { id: 'location' as Page, name: 'Location Analysis', icon: MapPin },
  ];

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage onNavigate={setCurrentPage} />;
      case 'rating':
        return <RatingPrediction />;
      case 'recommendations':
        return <Recommendations />;
      case 'cuisine':
        return <CuisineClassification />;
      case 'location':
        return <LocationAnalysis />;
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <nav className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Utensils className="h-8 w-8 text-orange-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">
                Restaurant Analysis
              </span>
            </div>

            <div className="hidden md:flex space-x-4">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => setCurrentPage(item.id)}
                    className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      currentPage === item.id
                        ? 'bg-orange-100 text-orange-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {item.name}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="md:hidden p-2 rounded-md text-gray-700 hover:bg-gray-100"
            >
              {menuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {menuOpen && (
          <div className="md:hidden border-t border-gray-200">
            <div className="px-2 pt-2 pb-3 space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setCurrentPage(item.id);
                      setMenuOpen(false);
                    }}
                    className={`flex items-center w-full px-3 py-2 rounded-md text-base font-medium ${
                      currentPage === item.id
                        ? 'bg-orange-100 text-orange-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="h-5 w-5 mr-3" />
                    {item.name}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderPage()}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-gray-600 text-sm">
            Cognifyz ML Internship Project - Restaurant Analysis and Explorer System
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
