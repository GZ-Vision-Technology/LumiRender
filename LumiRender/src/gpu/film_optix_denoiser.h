#pragma once

#include <core/film_denoiser.h>
#include <memory>

namespace luminous {

extern std::unique_ptr<FilmDenoiser> create_film_optix_denoiser();

};// namespace luminous
