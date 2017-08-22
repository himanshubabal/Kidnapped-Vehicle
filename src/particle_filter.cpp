#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>

#include "particle_filter.h"

using namespace std;

// * * * * OBSERVATIONS * * * *
// Number of particles     time-taken    error(x and y)
//      10                   65             .130, .160
//      50                   68.26          .122, .109
//      100                  69             .112, .104
//      1000                time-out!         - ,   -
//
//
//  FINAL CHOSEN - 100
//

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.

    default_random_engine gen;
    num_particles = 100;

    // Normal distributions for sensor data
    normal_distribution<double> Noise_x_init         (x, std[0]);
    normal_distribution<double> Noise_y_init         (y, std[1]);
    normal_distribution<double> Noise_theta_init (theta, std[2]);

    // Initialization of particles
    for (int i = 0; i < num_particles; i++){
        Particle particle;

        // Initial values of particle
        particle.id     = i;
        particle.weight = 1.0;

        particle.x     = Noise_x_init(gen);
        particle.y     = Noise_y_init(gen);
        particle.theta = Noise_theta_init(gen);

        particles.push_back(particle);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.

    default_random_engine gen;

    // Normal distributions for sensor noise
    // Gaussian with 0 mean and std_pos[] standard deviation
    normal_distribution<double> Noise_x    (0, std_pos[0]);
    normal_distribution<double> Noise_y    (0, std_pos[1]);
    normal_distribution<double> Noise_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++){
        // Two cases
        //  1. YawRate == 0
        //  2. YawRate != 0

        // Case YawRate == 0
        if(abs(yaw_rate) < 0.0001){
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
            // Theta won't change as yawrate == 0
        }
        else {
            particles[i].x     += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y     += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }

        // Gaussian Noise
        particles[i].x     += Noise_x(gen);
        particles[i].y     += Noise_y(gen);
        particles[i].theta += Noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.

    for (int i = 0; i < observations.size(); i++){
        LandmarkObs obs = observations[i];

        double min_distance = numeric_limits<double>::max();
        int id = -1;

        for (auto pred_obs : predicted) {
            // Find distance between predicted and observed points using inbuild method
            double distance = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
            if (distance < min_distance){
                min_distance = distance;
                id = pred_obs.id;
            }
        }
        observations[i].id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    for (int i = 0; i < num_particles; i++){
        double particle_x     = particles[i].x;
        double particle_y     = particles[i].y;
        double particle_theta = particles[i].theta;

        vector<LandmarkObs> predicted_landmarks;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j ++){
            int landmark_id   = map_landmarks.landmark_list[j].id_i;
            float landmark_x  = map_landmarks.landmark_list[j].x_f;
            float landmark_y  = map_landmarks.landmark_list[j].y_f;

            // Considering locations which are within sensor range
            if (dist(particle_x, particle_y, landmark_x, landmark_y) <= sensor_range){
                // add these to the predictions
                predicted_landmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
            }
        }

        // Transform observations from vehicle coordinate system to map coordinate system
        vector<LandmarkObs> transformed_observations;
        for (int j = 0; j < observations.size(); j++) {
            // x_new = x.cos(theta) - y.sin(theta) + x_old
            // y_new = x.sin(theta) + y.cos(theta) + y_old
            LandmarkObs obs = observations[j];
            double trans_x = obs.x * cos(particle_theta) - obs.y * sin(particle_theta) + particle_x;
            double trans_y = obs.x * sin(particle_theta) + obs.y * cos(particle_theta) + particle_y;

            transformed_observations.push_back(LandmarkObs{obs.id, trans_x, trans_y});
        }

        // Find closest predictions to landmarks
        dataAssociation(predicted_landmarks, transformed_observations);

        particles[i].weight = 1.0;
        for (int j = 0; j < transformed_observations.size(); j++){
            double obs_x  = transformed_observations[j].x;
            double obs_y  = transformed_observations[j].y;
            double obs_id = transformed_observations[j].id;

            // get closest particle to predicted location
            double pred_x = 0.0;
            double pred_y = 0.0;
            for (int k = 0; k < predicted_landmarks.size(); k++){
                if(predicted_landmarks[k].id == obs_id){
                    pred_x = predicted_landmarks[k].x;
                    pred_y = predicted_landmarks[k].y;
                }
            }

            // Calculate maultivariate gaussian weight
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];

            double c = 1/(2 * M_PI * std_x * std_y);
            double x_part = pow((obs_x - pred_x), 2)/(2 * pow(std_x, 2));
            double y_part = pow((obs_y - pred_y), 2)/(2 * pow(std_y, 2));
            double obs_w = c * exp(-(x_part + y_part));

            particles[i].weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.

    default_random_engine gen;
    vector<Particle> resampled_particles;

    // current weights
    vector<double> current_weights;
    current_weights.reserve(particles.size());
    for (int i = 0; i < particles.size(); i++){
        current_weights.push_back(particles[i].weight);
    }

    // generating random starting index for resampling wheel
    uniform_int_distribution<int> index_dist(0, num_particles - 1);
    int rand_index = index_dist(gen);

    // pointer to the maximum weight
    double w_max = *max_element(current_weights.begin(), current_weights.end());

    // uniform dist : range -> [0.0, w_max)
    uniform_real_distribution<double> uniform_weight_dist(0.0, w_max);

    double beta = 0.0;

    // spinning wheel
    for (int i = 0; i < num_particles; i++) {
        beta += uniform_weight_dist(gen) * 2.0;
        while (beta > current_weights[rand_index]) {
            beta -= current_weights[rand_index];
            rand_index = (rand_index + 1) % num_particles;
        }
        resampled_particles.push_back(particles[rand_index]);
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
