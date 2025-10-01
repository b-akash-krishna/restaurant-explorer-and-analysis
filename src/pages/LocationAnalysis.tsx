import { useState, useEffect } from 'react';
import { MapPin, Loader2, TrendingUp, Building2 } from 'lucide-react';

interface CityStats {
  city: string;
  count: number;
  avg_rating: number;
  avg_cost: number;
  top_cuisine: string;
}

interface LocalityStats {
  locality: string;
  city: string;
  count: number;
  avg_rating: number;
  price_range_mode: number;
}

interface Insights {
  total_restaurants: number;
  total_cities: number;
  total_localities: number;
  highest_rated_city: string;
  highest_restaurant_density_city: string;
  avg_rating_overall: number;
  most_expensive_city: string;
}

interface AnalysisData {
  insights: Insights;
  city_stats: CityStats[];
  locality_stats: LocalityStats[];
}

function LocationAnalysis() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<AnalysisData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'cities' | 'localities'>('overview');

  useEffect(() => {
    fetchAnalysis();
  }, []);

  const fetchAnalysis = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/location-analysis');
      if (!response.ok) throw new Error('Analysis failed');

      const result = await response.json();
      setData(result);
    } catch (err) {
      setError('Failed to load location analysis. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin h-8 w-8 text-red-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md">
        <p className="text-red-600 text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <MapPin className="h-8 w-8 text-red-600" />
        <h1 className="text-3xl font-bold text-gray-900">Location-based Analysis</h1>
      </div>

      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="border-b border-gray-200">
          <nav className="flex">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'overview'
                  ? 'border-b-2 border-red-600 text-red-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('cities')}
              className={`px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'cities'
                  ? 'border-b-2 border-red-600 text-red-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              City Analysis
            </button>
            <button
              onClick={() => setActiveTab('localities')}
              className={`px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'localities'
                  ? 'border-b-2 border-red-600 text-red-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Locality Analysis
            </button>
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'overview' && data?.insights && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Key Insights</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg">
                  <div className="text-sm text-blue-700 font-medium mb-1">Total Restaurants</div>
                  <div className="text-3xl font-bold text-blue-900">
                    {data.insights.total_restaurants.toLocaleString()}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
                  <div className="text-sm text-green-700 font-medium mb-1">Cities Covered</div>
                  <div className="text-3xl font-bold text-green-900">
                    {data.insights.total_cities}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg">
                  <div className="text-sm text-purple-700 font-medium mb-1">Localities</div>
                  <div className="text-3xl font-bold text-purple-900">
                    {data.insights.total_localities}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-lg">
                  <div className="text-sm text-yellow-700 font-medium mb-1">Average Rating</div>
                  <div className="text-3xl font-bold text-yellow-900">
                    {data.insights.avg_rating_overall.toFixed(2)}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-lg">
                  <div className="text-sm text-red-700 font-medium mb-1">Highest Rated City</div>
                  <div className="text-xl font-bold text-red-900">
                    {data.insights.highest_rated_city}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg">
                  <div className="text-sm text-orange-700 font-medium mb-1">Most Restaurants</div>
                  <div className="text-xl font-bold text-orange-900">
                    {data.insights.highest_restaurant_density_city}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'cities' && data?.city_stats && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <Building2 className="h-6 w-6 mr-2 text-red-600" />
                City Statistics
              </h2>
              <div className="space-y-3">
                {data.city_stats.map((city, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-semibold text-lg text-gray-900">{city.city}</h3>
                      <span className="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded">
                        {city.count} restaurants
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <div className="text-gray-500">Avg Rating</div>
                        <div className="font-medium text-gray-900">{city.avg_rating.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Avg Cost</div>
                        <div className="font-medium text-gray-900">{city.avg_cost.toFixed(0)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Top Cuisine</div>
                        <div className="font-medium text-gray-900 truncate">{city.top_cuisine}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'localities' && data?.locality_stats && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <TrendingUp className="h-6 w-6 mr-2 text-red-600" />
                Locality Statistics
              </h2>
              <div className="space-y-3">
                {data.locality_stats.map((locality, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="font-semibold text-gray-900">{locality.locality}</h3>
                        <p className="text-sm text-gray-500">{locality.city}</p>
                      </div>
                      <span className="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded">
                        {locality.count} restaurants
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm mt-3">
                      <div>
                        <div className="text-gray-500">Average Rating</div>
                        <div className="font-medium text-gray-900">{locality.avg_rating.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Price Range</div>
                        <div className="font-medium text-gray-900">
                          {'$'.repeat(locality.price_range_mode)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default LocationAnalysis;
