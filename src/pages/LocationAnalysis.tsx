import { useState, useEffect } from 'react';
import { MapPin, Loader2, TrendingUp, Building2, Globe, Search, Filter, BarChart3, PieChart, Map } from 'lucide-react';

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

interface MapLocation {
  lat: number;
  lng: number;
  name: string;
  city: string;
  rating: number;
}

interface AnalysisData {
  insights: Insights;
  city_stats: CityStats[];
  locality_stats: LocalityStats[];
}

function LocationAnalysis() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<AnalysisData | null>(null);
  const [mapData, setMapData] = useState<MapLocation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'cities' | 'localities' | 'map'>('overview');
  const [selectedCity, setSelectedCity] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'count' | 'rating' | 'cost'>('count');
  const [filterMinRating, setFilterMinRating] = useState(0);

  useEffect(() => {
    fetchAnalysis();
    fetchMapData();
  }, []);

  const fetchAnalysis = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/location-analysis');
      if (!response.ok) throw new Error('Analysis failed');

      const result = await response.json();
      setData(result);
      if (result.city_stats?.length > 0) {
        setSelectedCity(result.city_stats[0].city);
      }
    } catch (err) {
      setError('Failed to load location analysis');
    } finally {
      setLoading(false);
    }
  };

  const fetchMapData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/location-analysis/map');
      if (response.ok) {
        const result = await response.json();
        setMapData(result.locations || []);
      }
    } catch (err) {
      console.error('Failed to load map data');
    }
  };

  const loadLocalitiesForCity = async (city: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/location-analysis/localities/${city}`);
      if (response.ok) {
        const result = await response.json();
        setData(prev => prev ? { ...prev, locality_stats: result.localities } : null);
      }
    } catch (err) {
      console.error('Failed to load localities');
    }
  };

  useEffect(() => {
    if (selectedCity && activeTab === 'localities') {
      loadLocalitiesForCity(selectedCity);
    }
  }, [selectedCity, activeTab]);

  const getFilteredAndSortedCities = () => {
    if (!data?.city_stats) return [];
    
    let filtered = data.city_stats.filter(city => 
      city.city.toLowerCase().includes(searchTerm.toLowerCase()) &&
      city.avg_rating >= filterMinRating
    );

    filtered.sort((a, b) => {
      if (sortBy === 'count') return b.count - a.count;
      if (sortBy === 'rating') return b.avg_rating - a.avg_rating;
      return b.avg_cost - a.avg_cost;
    });

    return filtered;
  };

  const getFilteredLocalities = () => {
    if (!data?.locality_stats) return [];
    
    return data.locality_stats.filter(locality =>
      locality.locality.toLowerCase().includes(searchTerm.toLowerCase()) &&
      locality.avg_rating >= filterMinRating
    );
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

  const filteredCities = getFilteredAndSortedCities();
  const filteredLocalities = getFilteredLocalities();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Globe className="h-8 w-8 text-red-600 animate-pulse" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Location Analysis</h1>
            <p className="text-sm text-gray-500 mt-1">Interactive geographical insights and patterns</p>
          </div>
        </div>
      </div>

      {/* Overview Cards */}
      {data?.insights && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-6 rounded-xl shadow-lg text-white transform hover:scale-105 transition-transform">
            <div className="flex items-center justify-between mb-2">
              <Building2 className="h-8 w-8 opacity-80" />
              <div className="text-right">
                <div className="text-3xl font-bold">{data.insights.total_restaurants.toLocaleString()}</div>
                <div className="text-sm opacity-90">Restaurants</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-500 to-green-600 p-6 rounded-xl shadow-lg text-white transform hover:scale-105 transition-transform">
            <div className="flex items-center justify-between mb-2">
              <MapPin className="h-8 w-8 opacity-80" />
              <div className="text-right">
                <div className="text-3xl font-bold">{data.insights.total_cities}</div>
                <div className="text-sm opacity-90">Cities</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-500 to-purple-600 p-6 rounded-xl shadow-lg text-white transform hover:scale-105 transition-transform">
            <div className="flex items-center justify-between mb-2">
              <Map className="h-8 w-8 opacity-80" />
              <div className="text-right">
                <div className="text-3xl font-bold">{data.insights.total_localities}</div>
                <div className="text-sm opacity-90">Localities</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-yellow-500 to-orange-500 p-6 rounded-xl shadow-lg text-white transform hover:scale-105 transition-transform">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="h-8 w-8 opacity-80" />
              <div className="text-right">
                <div className="text-3xl font-bold">{data.insights.avg_rating_overall.toFixed(2)}</div>
                <div className="text-sm opacity-90">Avg Rating</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="border-b border-gray-200">
          <nav className="flex overflow-x-auto">
            {[
              { id: 'overview', label: 'Overview', icon: PieChart },
              { id: 'cities', label: 'City Analysis', icon: Building2 },
              { id: 'localities', label: 'Localities', icon: MapPin },
              { id: 'map', label: 'Map View', icon: Map }
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center px-6 py-4 text-sm font-medium transition-all whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'border-b-2 border-red-600 text-red-600 bg-red-50'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="w-4 h-4 mr-2" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'overview' && data?.insights && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Key Insights & Patterns</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-lg border border-red-200">
                  <div className="flex items-center mb-3">
                    <TrendingUp className="w-6 h-6 text-red-600 mr-2" />
                    <h3 className="font-semibold text-gray-900">Best Quality</h3>
                  </div>
                  <div className="text-2xl font-bold text-red-600 mb-1">{data.insights.highest_rated_city}</div>
                  <p className="text-sm text-gray-600">Highest average ratings</p>
                </div>

                <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-lg border border-orange-200">
                  <div className="flex items-center mb-3">
                    <Building2 className="w-6 h-6 text-orange-600 mr-2" />
                    <h3 className="font-semibold text-gray-900">Most Popular</h3>
                  </div>
                  <div className="text-2xl font-bold text-orange-600 mb-1">{data.insights.highest_restaurant_density_city}</div>
                  <p className="text-sm text-gray-600">Highest restaurant count</p>
                </div>

                <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border border-green-200">
                  <div className="flex items-center mb-3">
                    <BarChart3 className="w-6 h-6 text-green-600 mr-2" />
                    <h3 className="font-semibold text-gray-900">Premium Dining</h3>
                  </div>
                  <div className="text-2xl font-bold text-green-600 mb-1">{data.insights.most_expensive_city}</div>
                  <p className="text-sm text-gray-600">Highest average cost</p>
                </div>
              </div>

              {/* Distribution Chart Simulation */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-4">Restaurant Distribution</h3>
                <div className="space-y-3">
                  {data.city_stats?.slice(0, 5).map((city, index) => {
                    const maxCount = data.city_stats[0].count;
                    const percentage = (city.count / maxCount) * 100;
                    return (
                      <div key={city.city}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">{city.city}</span>
                          <span className="text-sm text-gray-600">{city.count} restaurants</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div
                            className="h-3 rounded-full transition-all duration-500"
                            style={{
                              width: `${percentage}%`,
                              background: `linear-gradient(90deg, ${
                                index === 0 ? '#ef4444, #dc2626' :
                                index === 1 ? '#f59e0b, #d97706' :
                                index === 2 ? '#10b981, #059669' :
                                index === 3 ? '#3b82f6, #2563eb' :
                                '#8b5cf6, #7c3aed'
                              })`
                            }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'cities' && (
            <div className="space-y-6">
              <div className="flex flex-col md:flex-row gap-4 items-end">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Search className="w-4 h-4 inline mr-1" />
                    Search Cities
                  </label>
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search by name..."
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Filter className="w-4 h-4 inline mr-1" />
                    Min Rating
                  </label>
                  <select
                    value={filterMinRating}
                    onChange={(e) => setFilterMinRating(Number(e.target.value))}
                    className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500"
                  >
                    <option value="0">Any</option>
                    <option value="3">3.0+</option>
                    <option value="3.5">3.5+</option>
                    <option value="4">4.0+</option>
                    <option value="4.5">4.5+</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500"
                  >
                    <option value="count">Restaurant Count</option>
                    <option value="rating">Average Rating</option>
                    <option value="cost">Average Cost</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredCities.map((city, index) => (
                  <div
                    key={city.city}
                    className="border border-gray-200 rounded-lg p-5 hover:shadow-xl transition-all hover:border-red-300 bg-gradient-to-br from-white to-gray-50"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-lg font-bold text-gray-900">{city.city}</span>
                          {index < 3 && (
                            <span className="bg-red-100 text-red-700 text-xs px-2 py-0.5 rounded-full font-semibold">
                              #{index + 1}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-500">{city.count} restaurants</p>
                      </div>
                      <div className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-bold">
                        ⭐ {city.avg_rating.toFixed(2)}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 mb-3">
                      <div className="bg-green-50 p-2 rounded">
                        <div className="text-xs text-gray-600">Avg Cost</div>
                        <div className="font-bold text-green-700">₹{city.avg_cost.toFixed(0)}</div>
                      </div>
                      <div className="bg-purple-50 p-2 rounded">
                        <div className="text-xs text-gray-600">Top Cuisine</div>
                        <div className="font-bold text-purple-700 text-xs truncate">{city.top_cuisine}</div>
                      </div>
                    </div>

                    <div className="pt-3 border-t border-gray-200">
                      <div className="flex items-center justify-between text-xs text-gray-600">
                        <span>Density</span>
                        <span className="font-semibold">{((city.count / data.insights.total_restaurants) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div
                          className="bg-gradient-to-r from-red-500 to-orange-500 h-2 rounded-full"
                          style={{ width: `${(city.count / data.city_stats[0].count) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {filteredCities.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                  <Search className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No cities found matching your criteria</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'localities' && (
            <div className="space-y-6">
              <div className="flex flex-col md:flex-row gap-4 items-end">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select City
                  </label>
                  <select
                    value={selectedCity}
                    onChange={(e) => setSelectedCity(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500"
                  >
                    {data?.city_stats?.map((city) => (
                      <option key={city.city} value={city.city}>
                        {city.city}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Search className="w-4 h-4 inline mr-1" />
                    Search Localities
                  </label>
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search..."
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Min Rating
                  </label>
                  <select
                    value={filterMinRating}
                    onChange={(e) => setFilterMinRating(Number(e.target.value))}
                    className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500"
                  >
                    <option value="0">Any</option>
                    <option value="3">3.0+</option>
                    <option value="3.5">3.5+</option>
                    <option value="4">4.0+</option>
                  </select>
                </div>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-sm text-blue-800">
                  <strong>Viewing {filteredLocalities.length} localities</strong> in {selectedCity}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredLocalities.map((locality, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-lg transition-all bg-white"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 mb-1">{locality.locality}</h3>
                        <p className="text-xs text-gray-500">{locality.city}</p>
                      </div>
                      <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded">
                        {locality.count}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-yellow-50 p-2 rounded">
                        <div className="text-xs text-gray-600">Rating</div>
                        <div className="font-semibold text-yellow-700 flex items-center">
                          {locality.avg_rating.toFixed(2)} ⭐
                        </div>
                      </div>
                      <div className="bg-green-50 p-2 rounded">
                        <div className="text-xs text-gray-600">Price</div>
                        <div className="font-semibold text-green-700">
                          {'₹'.repeat(locality.price_range_mode)}
                        </div>
                      </div>
                    </div>

                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-600">Restaurant Density</span>
                        <span className={`font-semibold ${
                          locality.count > 50 ? 'text-green-600' :
                          locality.count > 20 ? 'text-blue-600' : 'text-gray-600'
                        }`}>
                          {locality.count > 50 ? 'High' : locality.count > 20 ? 'Medium' : 'Low'}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {filteredLocalities.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                  <MapPin className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No localities found matching your criteria</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'map' && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <Map className="h-6 w-6 mr-2 text-red-600" />
                Restaurant Distribution Map
              </h2>

              {mapData.length > 0 ? (
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">📍 Location Data Loaded</h3>
                        <p className="text-sm text-gray-600 mt-1">
                          {mapData.length} restaurants with geographical coordinates
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold text-blue-600">{mapData.length}</div>
                        <div className="text-xs text-gray-500">Locations</div>
                      </div>
                    </div>

                    {/* Coordinate Range Info */}
                    <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-blue-200">
                      <div>
                        <div className="text-xs text-gray-600 mb-1">Latitude Range</div>
                        <div className="font-mono text-sm text-gray-900">
                          {Math.min(...mapData.map(l => l.lat)).toFixed(2)} to {Math.max(...mapData.map(l => l.lat)).toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-600 mb-1">Longitude Range</div>
                        <div className="font-mono text-sm text-gray-900">
                          {Math.min(...mapData.map(l => l.lng)).toFixed(2)} to {Math.max(...mapData.map(l => l.lng)).toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Search and Filter */}
                  <div className="flex gap-4">
                    <input
                      type="text"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      placeholder="Search restaurants or cities..."
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500"
                    />
                    <select
                      value={filterMinRating}
                      onChange={(e) => setFilterMinRating(Number(e.target.value))}
                      className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500"
                    >
                      <option value="0">All Ratings</option>
                      <option value="3">3.0+ ⭐</option>
                      <option value="4">4.0+ ⭐</option>
                      <option value="4.5">4.5+ ⭐</option>
                    </select>
                  </div>

                  {/* Map Locations Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto">
                    {mapData
                      .filter(loc => 
                        (loc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         loc.city.toLowerCase().includes(searchTerm.toLowerCase())) &&
                        loc.rating >= filterMinRating
                      )
                      .map((location, index) => (
                        <div
                          key={index}
                          className="border border-gray-200 rounded-lg p-3 hover:shadow-md transition-all bg-white hover:border-red-300"
                        >
                          <div className="flex items-start justify-between mb-2">
                            <h4 className="font-semibold text-sm text-gray-900 line-clamp-1">
                              {location.name}
                            </h4>
                            <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded flex-shrink-0 ml-2">
                              {location.rating} ⭐
                            </span>
                          </div>
                          <p className="text-xs text-gray-600 mb-2 flex items-center">
                            <MapPin className="w-3 h-3 mr-1" />
                            {location.city}
                          </p>
                          <div className="text-xs text-gray-500 font-mono bg-gray-50 p-1 rounded">
                            {location.lat.toFixed(4)}, {location.lng.toFixed(4)}
                          </div>
                        </div>
                      ))}
                  </div>

                  {/* Integration Suggestions */}
                  <div className="mt-6 p-6 bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg">
                    <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                      💡 Interactive Map Integration
                    </h4>
                    <p className="text-sm text-gray-700 mb-3">
                      To add a fully interactive map with markers, clustering, and click events, integrate one of these libraries:
                    </p>
                    <div className="space-y-2">
                      <div className="bg-white p-3 rounded border border-purple-100">
                        <strong className="text-sm text-purple-700">Leaflet (Recommended)</strong>
                        <p className="text-xs text-gray-600 mt-1">
                          Open-source, lightweight, mobile-friendly. Install: <code className="bg-gray-100 px-1">npm install react-leaflet leaflet</code>
                        </p>
                      </div>
                      <div className="bg-white p-3 rounded border border-purple-100">
                        <strong className="text-sm text-purple-700">Google Maps</strong>
                        <p className="text-xs text-gray-600 mt-1">
                          Rich features, requires API key. Install: <code className="bg-gray-100 px-1">npm install @react-google-maps/api</code>
                        </p>
                      </div>
                      <div className="bg-white p-3 rounded border border-purple-100">
                        <strong className="text-sm text-purple-700">Mapbox</strong>
                        <p className="text-xs text-gray-600 mt-1">
                          Beautiful custom styles. Install: <code className="bg-gray-100 px-1">npm install react-map-gl mapbox-gl</code>
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 bg-gray-50 rounded-lg">
                  <Map className="h-16 w-16 text-gray-300 mx-auto mb-3" />
                  <p className="text-gray-600">No map data available</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default LocationAnalysis;