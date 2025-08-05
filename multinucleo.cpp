#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>
#include <numeric>
#include <omp.h>
#include <algorithm>
#include <parallel/algorithm>
#include <map>
#include <utility>

// Configuration
const int width = 1920;
const int height = 1080;
const float plane_width = 400.0f;
const float plane_depth = 200.0f;
const int subdiv_x = 800;
const int subdiv_z = 200;

// Sky grid configuration
const float sky_height = 20.0f;
const int sky_subdiv_x = 300;
const int sky_subdiv_z = 100;
const float sky_plane_width = 1600.0f;
const float sky_plane_depth = 800.0f;
const cv::Scalar sky_color = cv::Scalar(255, 0, 0);

cv::Vec3f camera_position;
cv::Vec3f camera_rotation;
const float camera_base_x = 0.0f;
const float camera_base_y = 5.0f;
const float camera_base_z = -5.0f;
const float camera_amplitude_x = 0.05f;
const float camera_amplitude_y = 0.03f;
const float camera_amplitude_z = 0.05f;
const float camera_frequency_x = 0.02f;
const float camera_frequency_y = 0.015f;
const float camera_frequency_z = 0.01f;
const float rotation_amplitude_x = 0.12f;
const float rotation_amplitude_y = 0.16f;
const float rotation_amplitude_z = 0.08f;
const float rotation_frequency_x = 0.002f;
const float rotation_frequency_y = 0.0025f;
const float rotation_frequency_z = 0.0015f;
const float focal_length = 800.0f;

const float noise_scale = 0.1f;
const float noise_amplitude = 1.2f;
const int octaves = 4;
const float persistence = 0.5f;
const float lacunarity = 2.0f;
const float macro_noise_scale = 0.01f;
const float macro_noise_amplitude = 3.0f;

const float cloud_height = 5.0f;
const float cloud_noise_scale = 0.05f;
const float cloud_noise_amplitude = 1.0f;
const float cloud_scroll_speed = 0.01f;

const int max_circle_radius = 10;
const int min_circle_radius = 1;
const float max_screen_space_distance = 135.0f;
int frame_count = 0;
const float scroll_speed = 0.05f;

// Cloud threshold parameters
const float cloud_threshold_max = 1.0f;
const float cloud_threshold_min = 0.0f;
const float cloud_threshold_frequency = 0.001f;

std::vector<std::vector<cv::Vec2f>> base_grid;
std::vector<std::vector<cv::Vec2f>> sky_grid;

struct Dot {
    cv::Point pt;
    int radius;
    cv::Scalar color;
    float depth;
};

struct PointData {
    cv::Vec3f world_pos;
    cv::Point screen_pos;
    float theoretical_y;
    bool projected;
    bool is_water;
    bool is_cloud;
    bool is_sky;
};

const int NUM_THREADS = omp_get_max_threads();

void generate_base_grid() {
    base_grid.resize(subdiv_x + 1, std::vector<cv::Vec2f>(subdiv_z + 1));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= subdiv_x; ++i) {
        for (int j = 0; j <= subdiv_z; ++j) {
            float x = -plane_width / 2 + i * (plane_width / subdiv_x);
            float z = 1.0f + j * (plane_depth / subdiv_z);
            base_grid[i][j] = cv::Vec2f(x, z);
        }
    }
}

void generate_sky_grid() {
    sky_grid.resize(sky_subdiv_x + 1, std::vector<cv::Vec2f>(sky_subdiv_z + 1));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= sky_subdiv_x; ++i) {
        for (int j = 0; j <= sky_subdiv_z; ++j) {
            float x = -sky_plane_width / 2 + i * (sky_plane_width / sky_subdiv_x);
            float z = 1.0f + j * (sky_plane_depth / sky_subdiv_z);
            sky_grid[i][j] = cv::Vec2f(x, z);
        }
    }
}

int perm[512];

void init_permutation() {
    std::vector<int> p(256);
    std::iota(p.begin(), p.end(), 0);
    std::default_random_engine engine(42);
    std::shuffle(p.begin(), p.end(), engine);
    #pragma omp parallel for
    for (int i = 0; i < 512; ++i)
        perm[i] = p[i % 256];
}

float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float grad(int hash, float x, float y) {
    int h = hash & 3;
    float u = h < 2 ? x : y;
    float v = h < 2 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

float perlin(float x, float y) {
    int xi = (int)floor(x) & 255;
    int yi = (int)floor(y) & 255;
    float xf = x - floor(x);
    float yf = y - floor(y);
    float u = fade(xf);
    float v = fade(yf);
    int aa = perm[perm[xi] + yi];
    int ab = perm[perm[xi] + yi + 1];
    int ba = perm[perm[xi + 1] + yi];
    int bb = perm[perm[xi + 1] + yi + 1];
    float x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u);
    float x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u);
    return lerp(x1, x2, v);
}

float fractal_noise(float x, float y, float offset_x = 0.0f, float offset_y = 0.0f) {
    float total = 0.0f;
    float freq = 1.0f;
    float amp = 1.0f;
    float max_amp = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        total += perlin((x + offset_x) * freq, (y + offset_y) * freq) * amp;
        max_amp += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    return total / max_amp;
}

cv::Matx33f get_rotation_matrix(const cv::Vec3f& angles) {
    float cx = cos(angles[0]);
    float sx = sin(angles[0]);
    float cy = cos(angles[1]);
    float sy = sin(angles[1]);
    float cz = cos(angles[2]);
    float sz = sin(angles[2]);
    cv::Matx33f Rx(1, 0, 0,
                   0, cx, -sx,
                   0, sx, cx);
    cv::Matx33f Ry(cy, 0, sy,
                   0, 1, 0,
                   -sy, 0, cy);
    cv::Matx33f Rz(cz, -sz, 0,
                   sz, cz, 0,
                   0, 0, 1);
    return Rz * Ry * Rx;
}

bool project_point(const cv::Vec3f& point3D, cv::Point& projected) {
    cv::Matx33f R = get_rotation_matrix(camera_rotation);
    cv::Vec3f relative = point3D - camera_position;
    cv::Vec3f rotated = R * relative;
    float x = rotated[0], y = rotated[1], z = rotated[2];
    if (z <= 0.01f) return false;
    int u = static_cast<int>(width / 2 + focal_length * x / z);
    int v = static_cast<int>(height / 2 - focal_length * y / z);
    if (u < 0 || u >= width || v < 0 || v >= height) return false;
    projected = cv::Point(u, v);
    return true;
}

int compute_radius(float z) {
    float depth_min = 1.0f;
    float depth_max = plane_depth;
    float norm = std::clamp((z - depth_min) / (depth_max - depth_min), 0.0f, 1.0f);
    return std::round(max_circle_radius * (1.0f - norm) + min_circle_radius * norm);
}

struct ColorStop {
    float position;
    cv::Scalar color;
};

std::vector<ColorStop> color_gradient = {
    {-0.3f, cv::Scalar(50, 10, 10)},
    {-0.2f, cv::Scalar(100, 20, 20)},
    {-0.1f, cv::Scalar(150, 60, 30)},
    {-0.05f, cv::Scalar(200, 150, 100)},
    {0.00f, cv::Scalar(255, 220, 200)},
    {0.05f, cv::Scalar(80, 180, 50)},
    {0.2f, cv::Scalar(40, 120, 30)},
    {0.4f, cv::Scalar(80, 70, 60)},
    {0.6f, cv::Scalar(255, 255, 255)}};

cv::Scalar interpolate_color(const cv::Scalar& a, const cv::Scalar& b, float t) {
    return cv::Scalar(
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t
    );
}

cv::Scalar get_color_from_height(float y_normalized, bool is_water = false) {
    if (is_water) {
        for (size_t i = 1; i < color_gradient.size(); ++i) {
            if (y_normalized <= color_gradient[i].position) {
                float t = (y_normalized - color_gradient[i - 1].position) /
                          (color_gradient[i].position - color_gradient[i - 1].position);
                return interpolate_color(color_gradient[i - 1].color, color_gradient[i].color, t);
            }
        }
        return color_gradient.back().color;
    } else {
        for (size_t i = 1; i < color_gradient.size(); ++i) {
            if (y_normalized <= color_gradient[i].position) {
                float t = (y_normalized - color_gradient[i - 1].position) /
                          (color_gradient[i].position - color_gradient[i - 1].position);
                return interpolate_color(color_gradient[i - 1].color, color_gradient[i].color, t);
            }
        }
        return color_gradient.back().color;
    }
}

void adaptive_interpolation(std::vector<PointData>& points, std::vector<Dot>& dots, int max_iterations = 3) {
    // Create a spatial index that works with scrolling
    struct SpatialIndex {
        float cell_size;
        int grid_width;
        int grid_depth;
        std::vector<std::vector<PointData*>> grid;
        
        SpatialIndex(float width, float depth, float cell_sz) : 
            cell_size(cell_sz),
            grid_width(static_cast<int>(width/cell_sz) + 1),
            grid_depth(static_cast<int>(depth/cell_sz) + 1),
            grid(grid_width * grid_depth) {}
            
        void add_point(PointData* p) {
            int gx = static_cast<int>((p->world_pos[0] + plane_width/2) / cell_size);
            int gz = static_cast<int>((p->world_pos[2] - 1.0f) / cell_size);
            if (gx >= 0 && gx < grid_width && gz >= 0 && gz < grid_depth) {
                grid[gz * grid_width + gx].push_back(p);
            }
        }
        
        std::vector<PointData*> get_neighbors(int gx, int gz, int radius = 1) {
            std::vector<PointData*> result;
            for (int dz = -radius; dz <= radius; ++dz) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dz == 0) continue;
                    int nx = gx + dx;
                    int nz = gz + dz;
                    if (nx >= 0 && nx < grid_width && nz >= 0 && nz < grid_depth) {
                        result.insert(result.end(), grid[nz * grid_width + nx].begin(), 
                                    grid[nz * grid_width + nx].end());
                    }
                }
            }
            return result;
        }
    };

    float cell_size = plane_width / subdiv_x;
    SpatialIndex index(plane_width, plane_depth, cell_size);
    
    // Populate the spatial index
    for (auto& p : points) {
        if (p.projected) {
            index.add_point(&p);
        }
    }

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::vector<PointData> new_points;
        
        #pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i) {
            auto& p = points[i];
            if (!p.projected) continue;
            
            int gx = static_cast<int>((p.world_pos[0] + plane_width/2) / cell_size);
            int gz = static_cast<int>((p.world_pos[2] - 1.0f) / cell_size);
            
            auto neighbors = index.get_neighbors(gx, gz);
            
            for (auto neighbor : neighbors) {
                if (!neighbor->projected) continue;
                
                float screen_dist = cv::norm(p.screen_pos - neighbor->screen_pos);
                float world_dist = cv::norm(p.world_pos - neighbor->world_pos);
                
                // Adaptive distance threshold based on iteration
                float max_world_dist = cell_size * (2.0f + iteration);
                
                if (screen_dist > max_screen_space_distance && 
                    world_dist < max_world_dist) {
                    
                    // Calculate interpolation weight based on distance
                    float weight = 0.5f * (1.0f - (world_dist / max_world_dist));
                    
                    PointData mid;
                    mid.world_pos = p.world_pos * (1.0f - weight) + neighbor->world_pos * weight;
                    mid.theoretical_y = p.theoretical_y * (1.0f - weight) + neighbor->theoretical_y * weight;
                    mid.is_water = p.is_water && neighbor->is_water;
                    mid.is_cloud = p.is_cloud || neighbor->is_cloud;
                    mid.is_sky = p.is_sky || neighbor->is_sky;
                    mid.projected = project_point(mid.world_pos, mid.screen_pos);
                    
                    if (mid.projected) {
                        #pragma omp critical
                        new_points.push_back(mid);
                    }
                }
            }
        }
        
        // Add new points to the spatial index
        for (auto& p : new_points) {
            points.push_back(p);
            index.add_point(&points.back());
            
            if (iteration == max_iterations - 1) {
                Dot dot;
                dot.pt = p.screen_pos;
                dot.radius = compute_radius(p.world_pos[2] - camera_position[2]);
                float y_normalized = std::clamp(p.theoretical_y / (noise_amplitude + macro_noise_amplitude), -1.0f, 1.0f);
                
                if (p.is_cloud) {
                    dot.color = cv::Scalar(160, 160, 255);
                } else if (p.is_sky) {
                    dot.color = sky_color;
                } else {
                    dot.color = get_color_from_height(y_normalized, p.is_water);
                }
                
                dot.depth = p.world_pos[2] - camera_position[2];
                dots.push_back(dot);
            }
        }
        
        if (new_points.empty()) break;
    }
}

void render_terrain_with_adaptive_interpolation(float noise_offset_z, std::vector<Dot>& dots) {
    std::vector<PointData> points;
    
    // First collect all original grid points
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= subdiv_x; ++i) {
        for (int j = 0; j <= subdiv_z; ++j) {
            float x = base_grid[i][j][0];
            float z = base_grid[i][j][1];
            float rel_z = z - camera_position[2];
            if (rel_z <= 0.01f) continue;
            
            float world_z = z + noise_offset_z;
            float nx = x * noise_scale;
            float nz = world_z * noise_scale;
            float macro_nx = x * macro_noise_scale;
            float macro_nz = world_z * macro_noise_scale;
            
            float fine_noise = fractal_noise(nx, nz);
            float macro_noise = fractal_noise(macro_nx, macro_nz);
            float theoretical_y = fine_noise * noise_amplitude + macro_noise * macro_noise_amplitude;
            float actual_y = std::max(0.0f, theoretical_y);
            bool is_water = (theoretical_y < 0);
            
            PointData pd;
            pd.world_pos = cv::Vec3f(x, actual_y, z);
            pd.theoretical_y = theoretical_y;
            pd.is_water = is_water;
            pd.is_cloud = false;
            pd.is_sky = false;
            pd.projected = project_point(pd.world_pos, pd.screen_pos);
            
            if (pd.projected) {
                #pragma omp critical
                {
                    points.push_back(pd);
                    
                    // Add original point to dots
                    Dot dot;
                    dot.pt = pd.screen_pos;
                    dot.radius = compute_radius(rel_z);
                    float y_normalized = std::clamp(theoretical_y / (noise_amplitude + macro_noise_amplitude), -1.0f, 1.0f);
                    dot.color = get_color_from_height(y_normalized, is_water);
                    dot.depth = rel_z;
                    dots.push_back(dot);
                }
            }
        }
    }
    
    // Perform adaptive interpolation
    adaptive_interpolation(points, dots);
}

void render_sky_with_adaptive_interpolation(std::vector<Dot>& dots) {
    std::vector<PointData> points;
    
    // First collect all original grid points
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= sky_subdiv_x; ++i) {
        for (int j = 0; j <= sky_subdiv_z; ++j) {
            float x = sky_grid[i][j][0];
            float z = sky_grid[i][j][1];
            float rel_z = z - camera_position[2];
            if (rel_z <= 0.01f) continue;
            
            PointData pd;
            pd.world_pos = cv::Vec3f(x, sky_height, z);
            pd.theoretical_y = 0;
            pd.is_water = false;
            pd.is_cloud = false;
            pd.is_sky = true;
            pd.projected = project_point(pd.world_pos, pd.screen_pos);
            
            if (pd.projected) {
                #pragma omp critical
                {
                    points.push_back(pd);
                    
                    // Add original point to dots
                    Dot dot;
                    dot.pt = pd.screen_pos;
                    dot.radius = compute_radius(rel_z);
                    dot.color = sky_color;
                    dot.depth = rel_z;
                    dots.push_back(dot);
                }
            }
        }
    }
    
    // Perform adaptive interpolation
    adaptive_interpolation(points, dots);
}

void render_clouds_with_adaptive_interpolation(float noise_offset_z, float height, float noise_scale,
                                             float noise_amplitude, float noise_offset_x, 
                                             float noise_offset_y, float cloud_threshold,
                                             std::vector<Dot>& dots) {
    std::vector<PointData> points;
    
    // First collect all original grid points
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= subdiv_x; ++i) {
        for (int j = 0; j <= subdiv_z; ++j) {
            float x = base_grid[i][j][0];
            float z = base_grid[i][j][1];
            float rel_z = z - camera_position[2];
            if (rel_z <= 0.01f) continue;
            
            float world_z = z + noise_offset_z;
            float nx = x * noise_scale + noise_offset_x;
            float nz = world_z * noise_scale + noise_offset_y;
            float noise_val = fractal_noise(nx, nz);
            
            if (noise_val < cloud_threshold) continue;
            
            float y = height + (1.0f - noise_val) * noise_amplitude;
            
            PointData pd;
            pd.world_pos = cv::Vec3f(x, y, z);
            pd.theoretical_y = noise_val;
            pd.is_water = false;
            pd.is_cloud = true;
            pd.is_sky = false;
            pd.projected = project_point(pd.world_pos, pd.screen_pos);
            
            if (pd.projected) {
                #pragma omp critical
                {
                    points.push_back(pd);
                    
                    // Add original point to dots
                    Dot dot;
                    dot.pt = pd.screen_pos;
                    dot.radius = compute_radius(rel_z);
                    dot.color = cv::Scalar(160, 160, 255);
                    dot.depth = rel_z;
                    dots.push_back(dot);
                }
            }
        }
    }
    
    // Perform adaptive interpolation
    adaptive_interpolation(points, dots);
}

cv::Mat render_combined(float terrain_offset_z, float cloud_offset_z, float cloud_threshold) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<Dot> dots;
    
    // Render all components with adaptive interpolation
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            std::vector<Dot> terrain_dots;
            render_terrain_with_adaptive_interpolation(terrain_offset_z, terrain_dots);
            #pragma omp critical
            dots.insert(dots.end(), terrain_dots.begin(), terrain_dots.end());
        }
        
        #pragma omp section
        {
            std::vector<Dot> sky_dots;
            render_sky_with_adaptive_interpolation(sky_dots);
            #pragma omp critical
            dots.insert(dots.end(), sky_dots.begin(), sky_dots.end());
        }
        
        #pragma omp section
        {
            std::vector<Dot> cloud_dots1;
            render_clouds_with_adaptive_interpolation(cloud_offset_z, cloud_height, cloud_noise_scale,
                                                     cloud_noise_amplitude, 100.0f, 200.0f, 
                                                     cloud_threshold, cloud_dots1);
            #pragma omp critical
            dots.insert(dots.end(), cloud_dots1.begin(), cloud_dots1.end());
        }
        
        #pragma omp section
        {
            std::vector<Dot> cloud_dots2;
            render_clouds_with_adaptive_interpolation(cloud_offset_z * 0.6f, cloud_height + 2.0f, 
                                                     cloud_noise_scale * 0.7f, cloud_noise_amplitude * 1.2f,
                                                     300.0f, 400.0f, cloud_threshold, cloud_dots2);
            #pragma omp critical
            dots.insert(dots.end(), cloud_dots2.begin(), cloud_dots2.end());
        }
        
        #pragma omp section
        {
            std::vector<Dot> cloud_dots3;
            render_clouds_with_adaptive_interpolation(cloud_offset_z * 0.4f, cloud_height + 4.0f, 
                                                     cloud_noise_scale * 0.5f, cloud_noise_amplitude * 1.4f,
                                                     500.0f, 600.0f, cloud_threshold, cloud_dots3);
            #pragma omp critical
            dots.insert(dots.end(), cloud_dots3.begin(), cloud_dots3.end());
        }
    }
    
    // Sort all dots by depth (back to front)
    __gnu_parallel::sort(dots.begin(), dots.end(), [](const Dot& a, const Dot& b) {
        return a.depth > b.depth;
    });
    
    // Draw all dots - no parallel here to avoid OpenCV conflicts
    for (const auto& d : dots) {
        cv::circle(img, d.pt, d.radius, d.color, -1);
    }
    
    return img.clone(); // Ensure we return a unique copy
}

cv::Mat apply_glow(const cv::Mat& src) {
    cv::Mat blurred, result;
    cv::GaussianBlur(src, blurred, cv::Size(0, 0), 4);
    cv::add(src, blurred, result);
    return result;
}

int main() {
    omp_set_num_threads(NUM_THREADS);
    generate_base_grid();
    generate_sky_grid();
    init_permutation();
    
    time_t epoch = std::time(nullptr);
    std::string filename = "terrain_flythrough_" + std::to_string(epoch) + ".mp4";
    cv::VideoWriter writer;
    writer.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 60, cv::Size(width, height));
    
    if (!writer.isOpened()) {
        std::cerr << "Error opening video writer.\n";
        return -1;
    }
    
    const int total_frames = 60 * 60 * 60;
    while (frame_count < total_frames) {
        camera_position[0] = camera_base_x + camera_amplitude_x * sin(frame_count * camera_frequency_x);
        camera_position[1] = camera_base_y + camera_amplitude_y * sin(frame_count * camera_frequency_y);
        camera_position[2] = camera_base_z + camera_amplitude_z * sin(frame_count * camera_frequency_z);
        
        camera_rotation[0] = rotation_amplitude_x * sin(frame_count * rotation_frequency_x);
        camera_rotation[1] = rotation_amplitude_y * sin(frame_count * rotation_frequency_y);
        camera_rotation[2] = rotation_amplitude_z * sin(frame_count * rotation_frequency_z);
        
        float terrain_offset_z = frame_count * scroll_speed;
        float cloud_offset_z = frame_count * cloud_scroll_speed;
        float cloud_threshold = cloud_threshold_min + (cloud_threshold_max - cloud_threshold_min) * 
                              0.5f * (1.0f + sin(frame_count * cloud_threshold_frequency));
        
        cv::Mat base = render_combined(terrain_offset_z, cloud_offset_z, cloud_threshold);
        cv::Mat final = apply_glow(base);
        
        cv::imshow("Flythrough Terrain", final);
        writer.write(final);
        
        int key = cv::waitKey(1);
        if (key == 27) break;
        
        frame_count++;
    }
    
    writer.release();
    cv::destroyAllWindows();
    return 0;
}
