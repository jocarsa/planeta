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

// Configuración
int width = 1920;
int height = 1080;
float plane_width = 400.0f;
float plane_depth = 200.0f;
int subdiv_x = 800;
int subdiv_z = 200;

// Sky grid configuration
const float sky_height = 20.0f;
const int sky_subdiv_x = 150;
const int sky_subdiv_z = 50;
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
const float rotation_amplitude_x = 0.06f;
const float rotation_amplitude_y = 0.08f;
const float rotation_amplitude_z = 0.04f;
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
const int max_circle_radius = 4;
const int min_circle_radius = 1;
int frame_count = 0;
const float scroll_speed = 0.05f;

std::vector<std::vector<cv::Vec2f>> base_grid;
std::vector<std::vector<cv::Vec2f>> sky_grid;

struct Dot {
    cv::Point pt;
    int radius;
    cv::Scalar color;
    float depth;
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
    {0.6f, cv::Scalar(255, 255, 255)}
};

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
            if (y_normalized >= color_gradient[i-1].position && y_normalized < color_gradient[i].position) {
                float t = (y_normalized - color_gradient[i - 1].position) /
                          (color_gradient[i].position - color_gradient[i - 1].position);
                return interpolate_color(color_gradient[i - 1].color, color_gradient[i].color, t);
            }
        }
    }
    for (size_t i = 1; i < color_gradient.size(); ++i) {
        if (y_normalized <= color_gradient[i].position) {
            float t = (y_normalized - color_gradient[i - 1].position) /
                      (color_gradient[i].position - color_gradient[i - 1].position);
            return interpolate_color(color_gradient[i - 1].color, color_gradient[i].color, t);
        }
    }
    return color_gradient.back().color;
}

cv::Mat render_terrain(float noise_offset_z) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<std::vector<Dot>> thread_dots(NUM_THREADS);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for collapse(2) nowait
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
                
                float y_normalized = std::clamp(theoretical_y / (noise_amplitude + macro_noise_amplitude), -1.0f, 1.0f);
                cv::Vec3f point(x, actual_y, z);
                cv::Point proj;
                
                if (project_point(point, proj)) {
                    Dot dot;
                    dot.pt = proj;
                    dot.radius = compute_radius(rel_z);
                    dot.color = get_color_from_height(y_normalized, is_water);
                    dot.depth = rel_z;
                    thread_dots[thread_id].push_back(dot);
                }
            }
        }
    }

    std::vector<Dot> all_dots;
    for (auto& vec : thread_dots) {
        all_dots.insert(all_dots.end(), vec.begin(), vec.end());
    }

    __gnu_parallel::sort(all_dots.begin(), all_dots.end(), [](const Dot& a, const Dot& b) {
        return a.depth > b.depth;
    });

    #pragma omp parallel
    {
        cv::Mat thread_img = img.clone();
        #pragma omp for nowait
        for (size_t i = 0; i < all_dots.size(); ++i) {
            const auto& d = all_dots[i];
            cv::circle(thread_img, d.pt, d.radius, d.color, -1);
        }
        #pragma omp critical
        cv::add(img, thread_img, img);
    }

    return img;
}

void render_sky_grid_as_dots(cv::Mat& img, float noise_offset_z) {
    std::vector<std::vector<Dot>> thread_dots(NUM_THREADS);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for collapse(2) nowait
        for (int i = 0; i <= sky_subdiv_x; ++i) {
            for (int j = 0; j <= sky_subdiv_z; ++j) {
                float x = sky_grid[i][j][0];
                float z = sky_grid[i][j][1];
                float rel_z = z - camera_position[2];
                if (rel_z <= 0.01f) continue;

                cv::Vec3f point(x, sky_height, z);
                cv::Point proj;

                if (project_point(point, proj)) {
                    Dot dot;
                    dot.pt = proj;
                    dot.radius = compute_radius(rel_z);
                    dot.depth = rel_z;
                    thread_dots[thread_id].push_back(dot);
                }
            }
        }
    }

    std::vector<Dot> all_dots;
    for (auto& vec : thread_dots) {
        all_dots.insert(all_dots.end(), vec.begin(), vec.end());
    }
    __gnu_parallel::sort(all_dots.begin(), all_dots.end(), [](const Dot& a, const Dot& b) {
        return a.depth > b.depth;
    });

    #pragma omp parallel for
    for (size_t i = 0; i < all_dots.size(); ++i) {
        const auto& d = all_dots[i];
        #pragma omp critical
        cv::circle(img, d.pt, d.radius, sky_color, -1);
    }
}

void render_cloud_layer(cv::Mat& img, float noise_offset_z, float height, float noise_scale, 
                       float noise_amplitude, float noise_offset_x, float noise_offset_y) {
    std::vector<std::vector<Dot>> thread_dots(NUM_THREADS);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for collapse(2) nowait
        for (int i = 0; i <= subdiv_x; ++i) {
            for (int j = 0; j <= subdiv_z; ++j) {
                float x = base_grid[i][j][0];
                float z = base_grid[i][j][1];
                float rel_z = z - camera_position[2];
                if (rel_z <= 0.01f) continue;
                
                float world_z = z + noise_offset_z;
                float nx = x * noise_scale;
                float nz = world_z * noise_scale;
                float noise_val = fractal_noise(nx, nz, noise_offset_x, noise_offset_y);
                
                if (noise_val < 0.5f) continue;
                
                float y = height + (1.0f - noise_val) * noise_amplitude;
                cv::Vec3f point(x, y, z);
                cv::Point proj;
                
                if (project_point(point, proj)) {
                    Dot dot;
                    dot.pt = proj;
                    dot.radius = compute_radius(rel_z);
                    dot.depth = rel_z;
                    thread_dots[thread_id].push_back(dot);
                }
            }
        }
    }

    std::vector<Dot> all_dots;
    for (auto& vec : thread_dots) {
        all_dots.insert(all_dots.end(), vec.begin(), vec.end());
    }
    __gnu_parallel::sort(all_dots.begin(), all_dots.end(), [](const Dot& a, const Dot& b) {
        return a.depth > b.depth;
    });

    #pragma omp parallel for
    for (size_t i = 0; i < all_dots.size(); ++i) {
        const auto& d = all_dots[i];
        #pragma omp critical
        cv::circle(img, d.pt, d.radius, cv::Scalar(160, 160, 255), -1);
    }
}

cv::Mat render_combined(float terrain_offset_z, float cloud_offset_z) {
    cv::Mat img = render_terrain(terrain_offset_z);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        render_sky_grid_as_dots(img, terrain_offset_z);
        
        #pragma omp section
        render_cloud_layer(img, cloud_offset_z, cloud_height, cloud_noise_scale, 
                          cloud_noise_amplitude, 100.0f, 200.0f);
        
        #pragma omp section
        render_cloud_layer(img, cloud_offset_z * 0.6f, cloud_height + 2.0f, 
                          cloud_noise_scale * 0.7f, cloud_noise_amplitude * 1.2f, 
                          300.0f, 400.0f);
        
        #pragma omp section
        render_cloud_layer(img, cloud_offset_z * 0.4f, cloud_height + 4.0f, 
                          cloud_noise_scale * 0.5f, cloud_noise_amplitude * 1.4f, 
                          500.0f, 600.0f);
    }
    
    return img;
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
    cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 60, {width, height});
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
        
        cv::Mat base = render_combined(terrain_offset_z, cloud_offset_z);
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
