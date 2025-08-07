#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <omp.h>
#include <algorithm>
#include <deque>

// Configuration
const int width = 300;
const int height = 300;
const float plane_width = 400.0f;
const float plane_depth = 200.0f;
const int subdiv_x = 1600;
const int subdiv_z = 400;

// Terrain modification parameters
const float modification_radius = 200.0f;
const float modification_intensity = 200.0f;

// Modification layer
struct TerrainModification {
    float value = 0.0f;
    float persistence = 0.99f; // How long modifications last (per frame decay)
};

std::vector<std::vector<TerrainModification>> modification_layer(
    subdiv_x + 1, 
    std::vector<TerrainModification>(subdiv_z + 1)
);

// Anti-aliasing configuration
const int line_type = cv::LINE_AA;

// Glow configuration
const float glow_intensity = 0.05f;
const int glow_blur_size = 5;

// Fog configuration
const float fog_near = 20.0f;
const float fog_far = 400.0f;
const cv::Scalar fog_color = cv::Scalar(255, 255, 255);
const float fog_density = 1.0f;

// Sky fog configuration
const float sky_fog_near = 40.0f;
const float sky_fog_far = 400.0f;
const cv::Scalar sky_fog_color = cv::Scalar(255, 255, 255);
const float sky_fog_density = 1.0f;

// Sky grid configuration
const float sky_height = 20.0f;
const int sky_subdiv_x = 1200;
const int sky_subdiv_z = 300;
const float sky_plane_width = 1600.0f;
const float sky_plane_depth = 800.0f;
const cv::Scalar sky_color = cv::Scalar(255, 0, 0);

// Window background configuration
const cv::Scalar window_bg_color = cv::Scalar(255, 255, 255);

cv::Vec3f camera_position = cv::Vec3f(0.0f, 3.0f, 5.0f);
cv::Vec3f camera_rotation = cv::Vec3f(-0.1f, 0.0f, 0.0f);

// Camera movement speed
const float move_speed = 0.5f;
const float rotate_speed = 0.05f;

// Other constants and configurations remain the same
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
const float scroll_speed = 0.05f;

// Cloud threshold parameters
const float cloud_threshold_max = 1.0f;
const float cloud_threshold_min = 0.0f;
const float cloud_threshold_frequency = 0.001f;

// Tree parameters
const float max_tree_distance = 60.0f;
const int max_tree_depth = 4;
const float min_height_for_trees = 0.1f;
const float max_height_for_trees = 0.12f;
const float tree_size_multiplier = 0.2f;
const cv::Scalar tree_trunk_color = cv::Scalar(10, 30, 50);
const cv::Scalar tree_leaves_color = cv::Scalar(20, 60, 30);

// Motion blur configuration
const float motion_blur_strength = 0.9f;
const int motion_blur_frames = 9;

std::deque<cv::Mat> previous_frames;
std::vector<std::vector<cv::Vec2f>> base_grid;
std::vector<std::vector<cv::Vec2f>> sky_grid;

struct Dot {
    cv::Point pt;
    int radius;
    cv::Scalar color;
    float depth;
    float height;
    bool is_green;
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

template<typename T>
T clamp(const T& value, const T& low, const T& high) {
    return value < low ? low : (value > high ? high : value);
}

cv::Scalar apply_fog(const cv::Scalar& original_color, float distance, bool is_sky = false) {
    if (is_sky) {
        float fog_factor = 0.0f;
        if (distance > sky_fog_near) {
            float normalized_dist = (distance - sky_fog_near) / (sky_fog_far - sky_fog_near);
            normalized_dist = clamp(normalized_dist, 0.0f, 1.0f);
            fog_factor = 1.0f - exp(-sky_fog_density * normalized_dist * 5.0f);
        }
        cv::Scalar result;
        for (int i = 0; i < 3; i++) {
            result[i] = original_color[i] * (1.0f - fog_factor) + sky_fog_color[i] * fog_factor;
        }
        return result;
    } else {
        float fog_factor = 0.0f;
        if (distance > fog_near) {
            float normalized_dist = (distance - fog_near) / (fog_far - fog_near);
            normalized_dist = clamp(normalized_dist, 0.0f, 1.0f);
            fog_factor = 1.0f - exp(-fog_density * normalized_dist * 5.0f);
        }
        cv::Scalar result;
        for (int i = 0; i < 3; i++) {
            result[i] = original_color[i] * (1.0f - fog_factor) + fog_color[i] * fog_factor;
        }
        return result;
    }
}

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
    z = std::max(z, 1.0f);
    float normalized_z = z / plane_depth;
    float perspective_scale = 1.0f / normalized_z;
    float t = clamp(perspective_scale, 0.0f, 1.0f);
    return std::round(min_circle_radius + t * (max_circle_radius - min_circle_radius));
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

void draw_pythagorean_tree(cv::Mat& img, const cv::Point& base, float size, float angle, int depth,
                          const cv::Scalar& color, float size_multiplier, float tree_depth) {
    if (depth <= 0 || size < 0.5f) return;
    cv::Point tip;
    tip.x = base.x - static_cast<int>(size * sin(angle));
    tip.y = base.y - static_cast<int>(size * cos(angle));
    float t = static_cast<float>(max_tree_depth - depth) / max_tree_depth;
    cv::Scalar line_color = interpolate_color(
        apply_fog(tree_trunk_color, tree_depth),
        apply_fog(tree_leaves_color, tree_depth),
        t
    );
    int thickness = std::max(1, static_cast<int>(1.5f + depth * 0.5f));
    cv::line(img, base, tip, line_color, thickness, line_type);
    float size_reduction = 0.65f + 0.05f * (depth / static_cast<float>(max_tree_depth));
    float new_size = size * size_reduction;
    if (depth <= 2) {
        int leaf_size = std::max(1, static_cast<int>(2.0f + size * 0.3f));
        cv::circle(img, tip, leaf_size, apply_fog(tree_leaves_color, tree_depth), -1, line_type);
    }
    if (depth > 1) {
        float angle_variation = CV_PI / 5.0f * (0.8f + 0.4f * depth / static_cast<float>(max_tree_depth));
        draw_pythagorean_tree(img, tip, new_size, angle - angle_variation, depth - 1,
                            line_color, 1.0f, tree_depth);
        draw_pythagorean_tree(img, tip, new_size, angle + angle_variation, depth - 1,
                            line_color, 1.0f, tree_depth);
    }
}

void adaptive_interpolation(std::vector<PointData>& points, std::vector<Dot>& dots, int max_iterations = 3) {
    struct SpatialIndex {
        float cell_size;
        int grid_width;
        int grid_depth;
        std::vector<std::vector<PointData*>> grid;
        SpatialIndex(float width, float depth, float cell_sz) :
            cell_size(cell_sz),
            grid_width(static_cast<int>(width / cell_sz) + 1),
            grid_depth(static_cast<int>(depth / cell_sz) + 1),
            grid(grid_width * grid_depth) {}
        void add_point(PointData* p) {
            int gx = static_cast<int>((p->world_pos[0] + plane_width / 2) / cell_size);
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
            int gx = static_cast<int>((p.world_pos[0] + plane_width / 2) / cell_size);
            int gz = static_cast<int>((p.world_pos[2] - 1.0f) / cell_size);
            auto neighbors = index.get_neighbors(gx, gz);
            for (auto neighbor : neighbors) {
                if (!neighbor->projected) continue;
                float screen_dist = cv::norm(p.screen_pos - neighbor->screen_pos);
                float world_dist = cv::norm(p.world_pos - neighbor->world_pos);
                float max_world_dist = cell_size * (2.0f + iteration);
                if (screen_dist > max_screen_space_distance &&
                    world_dist < max_world_dist) {
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
        for (auto& p : new_points) {
            points.push_back(p);
            index.add_point(&points.back());
            if (iteration == max_iterations - 1) {
                Dot dot;
                dot.pt = p.screen_pos;
                dot.radius = compute_radius(p.world_pos[2] - camera_position[2]);
                float y_normalized = clamp(p.theoretical_y / (noise_amplitude + macro_noise_amplitude), -1.0f, 1.0f);
                dot.height = y_normalized;
                if (p.is_cloud) {
                    dot.color = apply_fog(cv::Scalar(160, 160, 255), dot.depth);
                    dot.is_green = false;
                } else if (p.is_sky) {
                    dot.color = apply_fog(sky_color, dot.depth, true);
                    dot.is_green = false;
                } else {
                    dot.color = apply_fog(get_color_from_height(y_normalized, p.is_water), dot.depth);
                    dot.is_green = (y_normalized >= min_height_for_trees && y_normalized <= max_height_for_trees);
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
            
            // Apply modification from our layer
            theoretical_y += modification_layer[i][j].value;
            
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
                    Dot dot;
                    dot.pt = pd.screen_pos;
                    dot.radius = compute_radius(rel_z);
                    float y_normalized = clamp(theoretical_y / (noise_amplitude + macro_noise_amplitude), -1.0f, 1.0f);
                    dot.height = y_normalized;
                    dot.color = apply_fog(get_color_from_height(y_normalized, is_water), rel_z);
                    dot.depth = rel_z;
                    dot.is_green = (y_normalized >= min_height_for_trees && y_normalized <= max_height_for_trees);
                    dots.push_back(dot);
                }
            }
        }
    }
    adaptive_interpolation(points, dots);
}

void render_sky_with_adaptive_interpolation(std::vector<Dot>& dots) {
    std::vector<PointData> points;
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
                    Dot dot;
                    dot.pt = pd.screen_pos;
                    dot.radius = compute_radius(rel_z);
                    dot.color = apply_fog(sky_color, rel_z, true);
                    dot.depth = rel_z;
                    dot.is_green = false;
                    dots.push_back(dot);
                }
            }
        }
    }
    adaptive_interpolation(points, dots);
}

void render_clouds_with_adaptive_interpolation(float noise_offset_z, float height, float noise_scale,
    float noise_amplitude, float noise_offset_x,
    float noise_offset_y, float cloud_threshold,
    std::vector<Dot>& dots) {
    std::vector<PointData> points;
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
                    Dot dot;
                    dot.pt = pd.screen_pos;
                    dot.radius = compute_radius(rel_z);
                    dot.color = apply_fog(cv::Scalar(160, 160, 255), rel_z);
                    dot.depth = rel_z;
                    dot.is_green = false;
                    dots.push_back(dot);
                }
            }
        }
    }
    adaptive_interpolation(points, dots);
}

cv::Mat render_combined(float terrain_offset_z, float cloud_offset_z, float cloud_threshold) {
    cv::Mat img(height, width, CV_8UC3, window_bg_color);
    std::vector<Dot> dots;
    std::vector<Dot> tree_dots;
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
    std::sort(dots.begin(), dots.end(), [](const Dot& a, const Dot& b) {
        return a.depth > b.depth;
    });
    for (const auto& d : dots) {
        cv::circle(img, d.pt, d.radius, d.color, -1, line_type);
        if (d.is_green && d.depth < max_tree_distance) {
            tree_dots.push_back(d);
        }
    }
    for (const auto& d : tree_dots) {
        float size_factor = 1.0f - (d.depth / max_tree_distance);
        int tree_size = static_cast<int>(8 + size_factor * 15);
        draw_pythagorean_tree(img, d.pt, tree_size * tree_size_multiplier, 0.0, max_tree_depth,
            tree_trunk_color, tree_size_multiplier, d.depth);
        cv::circle(img, d.pt, std::max(1, static_cast<int>(tree_size / 4 * tree_size_multiplier)),
            apply_fog(tree_trunk_color, d.depth), -1, line_type);
    }
    return img.clone();
}

cv::Mat apply_glow(const cv::Mat& src) {
    if (glow_intensity <= 0.0f) return src.clone();
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred,
        cv::Size(glow_blur_size, glow_blur_size),
        glow_blur_size * 0.6, 0,
        cv::BORDER_DEFAULT);
    cv::Mat result;
    cv::addWeighted(src, 1.0f - glow_intensity,
        blurred, glow_intensity,
        0.0, result);
    return result;
}

bool raycastTerrain(int screenX, int screenY, cv::Vec3f& worldPos) {
    float ndcX = (2.0f * screenX) / width - 1.0f;
    float ndcY = 1.0f - (2.0f * screenY) / height;

    cv::Matx33f R = get_rotation_matrix(camera_rotation);
    cv::Vec3f rayDir(ndcX, ndcY, -1.0f);
    rayDir = R.t() * rayDir;
    rayDir = cv::normalize(rayDir);

    float t = 0.0f;
    float step = 1.0f;
    while (t < 1000.0f) {
        cv::Vec3f point = camera_position + t * rayDir;
        if (point[2] < 1.0f || point[2] > plane_depth) {
            t += step;
            continue;
        }

        if (point[0] < -plane_width / 2 || point[0] > plane_width / 2) {
            t += step;
            continue;
        }

        float x = point[0];
        float z = point[2];
        float nx = x * noise_scale;
        float nz = z * noise_scale;
        float macro_nx = x * macro_noise_scale;
        float macro_nz = z * macro_noise_scale;
        float fine_noise = fractal_noise(nx, nz);
        float macro_noise = fractal_noise(macro_nx, macro_nz);
        float theoretical_y = fine_noise * noise_amplitude + macro_noise * macro_noise_amplitude;

        // Check modification layer
        int i = static_cast<int>((x + plane_width/2) / (plane_width/subdiv_x));
        int j = static_cast<int>((z - 1.0f) / (plane_depth/subdiv_z));
        if (i >= 0 && i <= subdiv_x && j >= 0 && j <= subdiv_z) {
            theoretical_y += modification_layer[i][j].value;
        }

        if (point[1] <= theoretical_y) {
            worldPos = point;
            return true;
        }

        t += step;
    }
    return false;
}

void modifyTerrain(int screenX, int screenY, bool elevate) {
    cv::Vec3f worldPos;
    if (raycastTerrain(screenX, screenY, worldPos)) {
        float strength = elevate ? modification_intensity : -modification_intensity;
        
        // Find the grid cell closest to the clicked world position
        int center_i = static_cast<int>((worldPos[0] + plane_width/2) / (plane_width/subdiv_x));
        int center_j = static_cast<int>((worldPos[2] - 1.0f) / (plane_depth/subdiv_z));
        
        // Calculate affected area
        int radius_cells = static_cast<int>(modification_radius / (plane_width/subdiv_x));
        
        #pragma omp parallel for collapse(2)
        for (int i = std::max(0, center_i-radius_cells); 
             i <= std::min(subdiv_x, center_i+radius_cells); ++i) {
            for (int j = std::max(0, center_j-radius_cells); 
                 j <= std::min(subdiv_z, center_j+radius_cells); ++j) {
                
                float x = base_grid[i][j][0];
                float z = base_grid[i][j][1];
                float dist = cv::norm(cv::Vec2f(x - worldPos[0], z - worldPos[2]));
                
                if (dist < modification_radius) {
                    float weight = strength * (1.0f - dist/modification_radius);
                    #pragma omp atomic
                    modification_layer[i][j].value += weight;
                }
            }
        }
    }
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        modifyTerrain(x, y, true);
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        modifyTerrain(x, y, false);
    }
}

int main() {
    omp_set_num_threads(NUM_THREADS);
    generate_base_grid();
    generate_sky_grid();
    init_permutation();

    const float motion_blur_amount = 0.3f;
    const int motion_blur_frames = 3;
    std::deque<cv::Mat> previous_frames;

    cv::namedWindow("Flythrough Terrain", cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
    cv::setMouseCallback("Flythrough Terrain", onMouse, nullptr);
    cv::resizeWindow("Flythrough Terrain", width, height);

    cv::Mat display_img;
    bool running = true;
    int frame_count = 0;

    while (running) {
        // Apply modification decay
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= subdiv_x; ++i) {
            for (int j = 0; j <= subdiv_z; ++j) {
                modification_layer[i][j].value *= modification_layer[i][j].persistence;
                if (fabs(modification_layer[i][j].value) < 0.01f) {
                    modification_layer[i][j].value = 0.0f;
                }
            }
        }

        int key = cv::waitKey(1);
        if (key == 'i') {
            camera_position[2] -= move_speed;
        } else if (key == 'k') {
            camera_position[2] += move_speed;
        } else if (key == 'j') {
            camera_position[0] -= move_speed;
        } else if (key == 'l') {
            camera_position[0] += move_speed;
        } else if (key == 'w') {
            camera_rotation[0] -= rotate_speed;
        } else if (key == 's') {
            camera_rotation[0] += rotate_speed;
        } else if (key == 'a') {
            camera_rotation[1] -= rotate_speed;
        } else if (key == 'd') {
            camera_rotation[1] += rotate_speed;
        } else if (key == 'c') { // Clear modifications
            #pragma omp parallel for collapse(2)
            for (int i = 0; i <= subdiv_x; ++i) {
                for (int j = 0; j <= subdiv_z; ++j) {
                    modification_layer[i][j].value = 0.0f;
                }
            }
        } else if (key == 27 || key == 'q') {
            running = false;
        }

        float terrain_offset_z = frame_count * scroll_speed;
        float cloud_offset_z = frame_count * cloud_scroll_speed;
        float cloud_threshold = cloud_threshold_min + (cloud_threshold_max - cloud_threshold_min) *
            0.5f * (1.0f + sin(frame_count * cloud_threshold_frequency));

        cv::Mat base = render_combined(terrain_offset_z, cloud_offset_z, cloud_threshold);
        cv::Mat final = apply_glow(base);

        if (motion_blur_amount > 0.0f && motion_blur_frames > 0) {
            cv::Mat blended;
            final.convertTo(blended, CV_32FC3, 1.0f - motion_blur_amount);
            float frame_weight = motion_blur_amount / motion_blur_frames;
            for (const auto& prev_frame : previous_frames) {
                cv::Mat temp;
                prev_frame.convertTo(temp, CV_32FC3, frame_weight);
                blended += temp;
            }
            blended.convertTo(final, CV_8UC3);
            if (previous_frames.size() >= motion_blur_frames) {
                previous_frames.pop_back();
            }
            previous_frames.push_front(base.clone());
        } else {
            previous_frames.clear();
        }

        if (cv::getWindowProperty("Flythrough Terrain", cv::WND_PROP_ASPECT_RATIO) > 0) {
            double scale = std::min(
                cv::getWindowImageRect("Flythrough Terrain").width / (double)final.cols,
                cv::getWindowImageRect("Flythrough Terrain").height / (double)final.rows
            );
            cv::resize(final, display_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
        } else {
            display_img = final;
        }

        cv::imshow("Flythrough Terrain", display_img);
        frame_count++;
    }

    cv::destroyAllWindows();
    return 0;
}
