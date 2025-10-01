import { TrendingUp, ThumbsUp, Utensils, MapPin, ArrowRight } from 'lucide-react';

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
    },
    {
      id: 'recommendations' as Page,
      title: 'Restaurant Recommendations',
      description: 'Get personalized restaurant recommendations based on cuisine preferences, location, and price range using content-based filtering.',
      icon: ThumbsUp,
      color: 'bg-green-500',
    },
    {
      id: 'cuisine' as Page,
      title: 'Cuisine Classification',
      description: 'Classify restaurant cuisines using multi-class classification models trained on restaurant features and characteristics.',
      icon: Utensils,
      color: 'bg-orange-500',
    },
    {
      id: 'location' as Page,
      title: 'Location-based Analysis',
      description: 'Explore geospatial visualizations and analyze restaurant distributions, ratings, and trends across different cities and localities.',
      icon: MapPin,
      color: 'bg-red-500',
    },
  ];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Restaurant ML Analysis Platform
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Comprehensive machine learning solutions for restaurant data analysis. Explore rating predictions,
          personalized recommendations, cuisine classification, and location-based insights.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-12">
        {tasks.map((task) => {
          const Icon = task.icon;
          return (
            <div
              key={task.id}
              className="bg-white rounded-lg shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden group cursor-pointer"
              onClick={() => onNavigate(task.id)}
            >
              <div className={`${task.color} h-2`}></div>
              <div className="p-6">
                <div className="flex items-start justify-between">
                  <div className={`${task.color} p-3 rounded-lg`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <ArrowRight className="h-5 w-5 text-gray-400 group-hover:text-gray-600 group-hover:translate-x-1 transition-all" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mt-4 mb-2">
                  {task.title}
                </h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {task.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-white rounded-lg shadow-md p-8 mt-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">About This Project</h2>
        <div className="prose prose-slate max-w-none">
          <p className="text-gray-600 mb-4">
            This platform is part of the Cognifyz Technologies Machine Learning Internship Program.
            It demonstrates practical applications of various ML techniques on real-world restaurant data.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Technologies Used</h3>
              <ul className="text-gray-600 space-y-1 text-sm">
                <li>Backend: Python, FastAPI</li>
                <li>Frontend: React, TypeScript, Tailwind CSS</li>
                <li>ML: Scikit-learn, Pandas, NumPy</li>
                <li>Deployment: Docker, GitHub Actions</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">ML Techniques</h3>
              <ul className="text-gray-600 space-y-1 text-sm">
                <li>Regression Analysis</li>
                <li>Content-based Filtering</li>
                <li>Multi-class Classification</li>
                <li>Geospatial Analysis</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
