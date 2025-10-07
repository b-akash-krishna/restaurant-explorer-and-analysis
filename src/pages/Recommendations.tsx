import { useState, useEffect } from 'react';
import { ThumbsUp, Loader2, RefreshCw, Star, DollarSign, Users, MapPin, Utensils, CheckCircle, XCircle, Sparkles, TrendingUp } from 'lucide-react';

interface Restaurant {
  name: string;
  cuisine: string;
  city: string;
  rating: number;
  cost: number;
  votes: number;
  price_range: number;
  online_delivery: string;
  table_booking: string;
  similarity_score: number;
}

interface RecommendationOptions {
  cuisines: string[];
  cities: string[];
}

function Recommendations() {
  const [loading, setLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<Restaurant[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<RecommendationOptions | null>(null);

  const [filters, setFilters] = useState({
    cuisine: '',
    location: '',
    count: 10,
    price_range: null as number | null,
    table_booking: null as boolean | null,
    online_delivery: null as boolean | null,
  });

  useEffect(() => {
    fetchOptions();
  }, []);

  const fetchOptions = async () => {
    setOptionsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/recommend-restaurants/options');
      if (response.ok) {
        const data = await response.json();
        setOptions(data);
        if (data.cuisines?.length > 0) {
          setFilters(prev => ({ ...prev, cuisine: data.cuisines[0] }));
        }
        if (data.cities?.length > 0) {
          setFilters(prev => ({ ...prev, location: data.cities[0] }));
        }
      }
    } catch (err) {
      setError('Failed to load options');
    } finally {
      setOptionsLoading(false);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setRecommendations([]);

    try {
      const response = await fetch('http://localhost:8000/api/recommend-restaurants', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cuisine: filters.cuisine,
          location: filters.location,
          count: filters.count,
          price_range: filters.price_range,
          table_booking: filters.table_booking,
          online_delivery: filters.online_delivery,
        }),
      });

      if (!response.ok) throw new Error('Recommendation failed');

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      setError('Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const resetFilters = () => {
    setFilters({
      cuisine: options?.cuisines[0] || '',
      location: options?.cities[0] || '',
      count: 10,
      price_range: null,
      table_booking: null,
      online_delivery: null,
    });
    setRecommendations([]);
  };

  const getPriceRangeLabel = (range: number) => {
    const labels = ['', 'Budget (₹)', 'Moderate (₹₹)', 'Expensive (₹₹₹)', 'Very Expensive (₹₹₹₹)'];
    return labels[range] || 'Unknown';
  };

  if (optionsLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 flex items-center justify-center">
        <Loader2 className="animate-spin h-12 w-12 text-green-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      {/* Hero Header */}
      <div 
        className="relative bg-cover bg-center py-16 mb-8"
        style={{
          backgroundImage: "url('https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=1600&h=400&fit=crop')",
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-green-900/90 via-emerald-800/85 to-teal-900/90"></div>
        <div className="relative max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-white/20 backdrop-blur-md p-4 rounded-2xl">
              <Sparkles className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-center text-white mb-3 drop-shadow-lg">
            Restaurant Recommendations
          </h1>
          <p className="text-center text-green-100 text-lg max-w-2xl mx-auto drop-shadow-md">
            Discover your perfect dining experience with personalized AI-powered recommendations
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 pb-8 space-y-6">
        {/* Filter Panel */}
        <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-green-100">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Users className="h-6 w-6 mr-3 text-green-600" />
            Set Your Preferences
          </h2>

          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="flex items-center text-sm font-semibold text-gray-700 mb-2">
                  <MapPin className="w-4 h-4 mr-2 text-green-600" />
                  City / Location
                </label>
                <select
                  value={filters.location}
                  onChange={(e) => setFilters(prev => ({ ...prev, location: e.target.value }))}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-green-100 focus:border-green-500 transition-all duration-200"
                >
                  {options?.cities.map((city) => (
                    <option key={city} value={city}>{city}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="flex items-center text-sm font-semibold text-gray-700 mb-2">
                  <Utensils className="w-4 h-4 mr-2 text-green-600" />
                  Cuisine Type
                </label>
                <select
                  value={filters.cuisine}
                  onChange={(e) => setFilters(prev => ({ ...prev, cuisine: e.target.value }))}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-green-100 focus:border-green-500 transition-all duration-200"
                >
                  {options?.cuisines.map((cuisine) => (
                    <option key={cuisine} value={cuisine}>{cuisine}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="flex items-center text-sm font-semibold text-gray-700 mb-2">
                  <DollarSign className="w-4 h-4 mr-2 text-green-600" />
                  Price Range
                </label>
                <select
                  value={filters.price_range === null ? '' : filters.price_range}
                  onChange={(e) => setFilters(prev => ({ 
                    ...prev, 
                    price_range: e.target.value === '' ? null : Number(e.target.value) 
                  }))}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-green-100 focus:border-green-500 transition-all duration-200"
                >
                  <option value="">Any Price</option>
                  <option value="1">Budget (₹)</option>
                  <option value="2">Moderate (₹₹)</option>
                  <option value="3">Expensive (₹₹₹)</option>
                  <option value="4">Very Expensive (₹₹₹₹)</option>
                </select>
              </div>

              <div>
                <label className="flex items-center text-sm font-semibold text-gray-700 mb-2">
                  <CheckCircle className="w-4 h-4 mr-2 text-purple-600" />
                  Table Booking
                </label>
                <select
                  value={filters.table_booking === null ? '' : String(filters.table_booking)}
                  onChange={(e) => setFilters(prev => ({ 
                    ...prev, 
                    table_booking: e.target.value === '' ? null : e.target.value === 'true' 
                  }))}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-green-100 focus:border-green-500 transition-all duration-200"
                >
                  <option value="">Any</option>
                  <option value="true">Available</option>
                  <option value="false">Not Required</option>
                </select>
              </div>

              <div>
                <label className="flex items-center text-sm font-semibold text-gray-700 mb-2">
                  <TrendingUp className="w-4 h-4 mr-2 text-orange-600" />
                  Online Delivery
                </label>
                <select
                  value={filters.online_delivery === null ? '' : String(filters.online_delivery)}
                  onChange={(e) => setFilters(prev => ({ 
                    ...prev, 
                    online_delivery: e.target.value === '' ? null : e.target.value === 'true' 
                  }))}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-green-100 focus:border-green-500 transition-all duration-200"
                >
                  <option value="">Any</option>
                  <option value="true">Available</option>
                  <option value="false">Not Required</option>
                </select>
              </div>
            </div>

            <div>
              <label className="flex items-center text-sm font-semibold text-gray-700 mb-2">
                <Users className="w-4 h-4 mr-2 text-green-600" />
                Number of Results
              </label>
              <select
                value={filters.count}
                onChange={(e) => setFilters(prev => ({ ...prev, count: Number(e.target.value) }))}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-green-100 focus:border-green-500 transition-all duration-200"
              >
                <option value="5">5 restaurants</option>
                <option value="10">10 restaurants</option>
                <option value="15">15 restaurants</option>
                <option value="20">20 restaurants</option>
              </select>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 text-white py-4 rounded-xl font-semibold hover:from-green-700 hover:to-emerald-700 transition-all duration-300 disabled:from-green-300 disabled:to-emerald-300 disabled:cursor-not-allowed flex items-center justify-center shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5 mr-2" />
                    Finding Restaurants...
                  </>
                ) : (
                  <>
                    <ThumbsUp className="h-5 w-5 mr-2" />
                    Get Recommendations
                  </>
                )}
              </button>
              <button
                onClick={resetFilters}
                className="px-8 py-4 bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 rounded-xl font-semibold transition-all duration-300 flex items-center shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Reset
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-xl">
              <p className="text-red-600 text-sm font-medium">{error}</p>
            </div>
          )}
        </div>

        {/* Top Recommendations Highlight */}
        {recommendations.length > 0 && (
          <>
            <div className="bg-gradient-to-br from-green-500 via-emerald-500 to-teal-500 rounded-2xl shadow-2xl p-8 border-2 border-green-300">
              <h2 className="text-3xl font-bold text-white mb-4 flex items-center">
                <Sparkles className="h-8 w-8 mr-3" />
                Top Recommendations for You
              </h2>
              <p className="text-green-50 text-sm mb-6">
                Based on: <span className="font-bold">{filters.cuisine}</span> in <span className="font-bold">{filters.location}</span>
                {filters.price_range && <>, {getPriceRangeLabel(filters.price_range)}</>}
                {filters.table_booking !== null && <>, Table Booking: {filters.table_booking ? 'Required' : 'Not Required'}</>}
                {filters.online_delivery !== null && <>, Online Delivery: {filters.online_delivery ? 'Required' : 'Not Required'}</>}
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {recommendations.slice(0, 6).map((restaurant, index) => (
                  <div
                    key={index}
                    className="bg-white/95 backdrop-blur-sm rounded-xl p-5 shadow-lg hover:shadow-2xl transition-all duration-300 border-l-4 border-amber-400 transform hover:-translate-y-1"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-gray-900 mb-1 line-clamp-1">
                          {restaurant.name}
                        </h3>
                        <p className="text-xs text-gray-600 flex items-center">
                          <Utensils className="w-3 h-3 mr-1" />
                          {restaurant.cuisine.split(',')[0]}
                        </p>
                      </div>
                      <div className="bg-amber-100 text-amber-800 px-3 py-1 rounded-full font-bold text-sm flex items-center shrink-0">
                        <Star className="w-4 h-4 mr-1 fill-amber-500" />
                        {restaurant.rating.toFixed(1)}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex items-center text-sm text-gray-600">
                        <MapPin className="w-4 h-4 mr-1 text-green-600" />
                        <span className="font-medium">{restaurant.city}</span>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center text-green-700 font-bold">
                          <DollarSign className="w-4 h-4 mr-1" />
                          ₹{restaurant.cost}
                        </div>
                        <div className="text-gray-500 text-xs">
                          {restaurant.votes} votes
                        </div>
                      </div>

                      <div className="flex gap-2 flex-wrap">
                        <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                          restaurant.price_range <= 2 ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
                        }`}>
                          {getPriceRangeLabel(restaurant.price_range)}
                        </span>
                        {restaurant.online_delivery?.toLowerCase() === 'yes' && (
                          <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded-full font-medium">
                            🚚 Delivery
                          </span>
                        )}
                        {restaurant.table_booking?.toLowerCase() === 'yes' && (
                          <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded-full font-medium">
                            📅 Booking
                          </span>
                        )}
                      </div>

                      {restaurant.similarity_score > 0 && (
                        <div className="pt-3 border-t border-gray-200">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-600 font-medium">Match Score</span>
                            <span className="font-bold text-green-600">
                              {(restaurant.similarity_score * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div
                              className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${restaurant.similarity_score * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* All Recommendations */}
            <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-green-100">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-2xl font-bold text-gray-900">
                  All {recommendations.length} Recommendations
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {recommendations.map((restaurant, index) => (
                  <div
                    key={index}
                    className="border-2 border-gray-200 rounded-xl p-6 hover:shadow-xl transition-all duration-300 hover:border-green-300 bg-gradient-to-br from-white to-green-50/30 transform hover:-translate-y-1"
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div className="flex-1">
                        <h3 className="font-bold text-xl text-gray-900 mb-2">{restaurant.name}</h3>
                        <p className="text-sm text-gray-600 flex items-center">
                          <Utensils className="w-4 h-4 mr-1" />
                          {restaurant.cuisine}
                        </p>
                      </div>
                      <div className="bg-amber-100 text-amber-800 px-4 py-2 rounded-full font-bold text-base flex items-center">
                        <Star className="w-5 h-5 mr-1 fill-amber-500" />
                        {restaurant.rating.toFixed(1)}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-green-50 p-3 rounded-lg">
                        <div className="text-xs text-gray-600 mb-1">Location</div>
                        <div className="font-semibold text-sm text-green-700 flex items-center">
                          <MapPin className="w-4 h-4 mr-1" />
                          {restaurant.city}
                        </div>
                      </div>
                      <div className="bg-blue-50 p-3 rounded-lg">
                        <div className="text-xs text-gray-600 mb-1">Cost for Two</div>
                        <div className="font-semibold text-sm text-blue-700 flex items-center">
                          <DollarSign className="w-4 h-4 mr-1" />
                          ₹{restaurant.cost}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-4 border-t border-gray-200">
                      <div className="flex gap-2 flex-wrap">
                        <span className={`text-xs px-3 py-1 rounded-full font-semibold ${
                          restaurant.price_range <= 2 ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
                        }`}>
                          {getPriceRangeLabel(restaurant.price_range)}
                        </span>
                        {restaurant.online_delivery?.toLowerCase() === 'yes' ? (
                          <span className="bg-purple-100 text-purple-700 text-xs px-3 py-1 rounded-full font-semibold flex items-center">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Delivery
                          </span>
                        ) : (
                          <span className="bg-gray-100 text-gray-600 text-xs px-3 py-1 rounded-full font-semibold flex items-center">
                            <XCircle className="w-3 h-3 mr-1" />
                            No Delivery
                          </span>
                        )}
                        {restaurant.table_booking?.toLowerCase() === 'yes' ? (
                          <span className="bg-blue-100 text-blue-700 text-xs px-3 py-1 rounded-full font-semibold flex items-center">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Booking
                          </span>
                        ) : (
                          <span className="bg-gray-100 text-gray-600 text-xs px-3 py-1 rounded-full font-semibold flex items-center">
                            <XCircle className="w-3 h-3 mr-1" />
                            No Booking
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 font-medium">
                        {restaurant.votes} votes
                      </div>
                    </div>

                    {restaurant.similarity_score > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <div className="flex items-center justify-between text-xs mb-2">
                          <span className="text-gray-600 font-medium">Match Score</span>
                          <span className="font-bold text-green-600">
                            {(restaurant.similarity_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${restaurant.similarity_score * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Empty State */}
        {!loading && recommendations.length === 0 && !error && (
          <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-16 text-center border border-green-100">
            <Sparkles className="h-20 w-20 text-green-300 mx-auto mb-6" />
            <h3 className="text-2xl font-bold text-gray-700 mb-3">
              Ready to Discover?
            </h3>
            <p className="text-gray-500 text-lg">
              Set your preferences and click "Get Recommendations" to find your perfect restaurant
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Recommendations;