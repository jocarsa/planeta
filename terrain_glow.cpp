#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>

// === Framebuffer size ===
const int width = 1920;
const int height = 1080;

// === Plane and grid parameters ===
const float plane_width = 400.0f;
const float plane_depth = 200.0f;
const int subdiv_x = 1600;
const int subdiv_z = 400;

// === Camera parameters ===
cv::Vec3f camera_position(0.0f, 3.0f, -2.0f);
const float focal_length = 800.0f;

// === Noise parameters ===
const float noise_scale = 0.1f;
const float noise_amplitude = 1.2f;
const int octaves = 4;
const float persistence = 0.5f;
const float lacunarity = 2.0f;

// === Animation parameters ===
int frame_count = 0;
const float scroll_speed = 0.05f;

// === Base grid ===
std::vector<std::vector<cv::Vec2f>> base_grid;

void generate_base_grid() {
    base_grid.resize(subdiv_x + 1, std::vector<cv::Vec2f>(subdiv_z + 1));
    for (int i = 0; i <= subdiv_x; ++i) {
        for (int j = 0; j <= subdiv_z; ++j) {
            float x = -plane_width / 2 + i * (plane_width / subdiv_x);
            float z = 1.0f + j * (plane_depth / subdiv_z);
            base_grid[i][j] = cv::Vec2f(x, z);
        }
    }
}

// === Pure C++ Perlin Noise ===
int perm[512];
void init_permutation() {
    std::vector<int> p(256);
    std::iota(p.begin(), p.end(), 0);
    std::default_random_engine engine(42);
    std::shuffle(p.begin(), p.end(), engine);
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

float fractal_noise(float x, float y) {
    float total = 0.0f;
    float freq = 1.0f;
    float amp = 1.0f;
    float max_amp = 0.0f;

    for (int i = 0; i < octaves; ++i) {
        total += perlin(x * freq, y * freq) * amp;
        max_amp += amp;
        amp *= persistence;
        freq *= lacunarity;
    }

    return total / max_amp;
}

// === Projection ===
cv::Point project_point(const cv::Vec3f& point3D) {
    cv::Vec3f relative = point3D - camera_position;
    float x = relative[0], y = relative[1], z = relative[2];
    if (z <= 0.01f) return {-1, -1};
    int u = static_cast<int>(width / 2 + focal_length * x / z);
    int v = static_cast<int>(height / 2 - focal_length * y / z);
    return (u >= 0 && u < width && v >= 0 && v < height) ? cv::Point(u, v) : cv::Point(-1, -1);
}

// === Render and glow ===
cv::Mat render(float noise_offset_z) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i <= subdiv_x; ++i) {
        for (int j = 0; j <= subdiv_z; ++j) {
            float x = base_grid[i][j][0];
            float z = base_grid[i][j][1];
            float nx = i * noise_scale;
            float nz = j * noise_scale + noise_offset_z;
            float y = fractal_noise(nx, nz) * noise_amplitude;
            cv::Vec3f point(x, y, z);
            cv::Point proj = project_point(point);
            if (proj.x != -1)
                cv::circle(img, proj, 1, cv::Scalar(255, 255, 255), -1);
        }
    }
    return img;
}

cv::Mat apply_glow(const cv::Mat& src) {
    cv::Mat blurred, result;
    cv::GaussianBlur(src, blurred, cv::Size(0, 0), 4);
    cv::add(src, blurred, result);
    return result;
}

// === Main ===
int main() {
    generate_base_grid();
    init_permutation();

    time_t epoch = std::time(nullptr);
    std::string filename = "terrain_" + std::to_string(epoch) + ".mp4";
    cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 60, {width, height});
    if (!writer.isOpened()) {
        std::cerr << "❌ Error opening video writer.\n";
        return -1;
    }

    const int total_frames = 60 * 60 * 60;  // 60 min × 60 sec × 60 fps = 216000

    while (frame_count < total_frames) {
        float offset_z = frame_count * scroll_speed;
        cv::Mat base = render(offset_z);
        cv::Mat final = apply_glow(base);

        cv::imshow("Fractal Terrain Animation", final);
        writer.write(final);

        int key = cv::waitKey(1);
        if (key == 27) break; // ESC para salir manualmente

        frame_count++;
    }

    writer.release();
    cv::destroyAllWindows();
    return 0;
}


