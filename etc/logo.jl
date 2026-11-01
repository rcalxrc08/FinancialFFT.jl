# Draws etc/logo.png and etc/favicon.ico, in the style of the FinancialToolbox.jl logo: three discs in the Julia colours.
# Needs Luxor.jl, ForwardDiff.jl, FinancialFFT.jl, FinancialMonteCarlo.jl and FinancialToolbox.jl in the active environment: julia etc/logo.jl
using Luxor
import ForwardDiff
using FinancialFFT: CarrMadanMethod, characteristic_exponent_i
using FinancialMonteCarlo: EuropeanOptionSmile, HestonProcess, MertonProcess, Underlying, ZeroRate, pricer
using FinancialToolbox: blsimpv

const WIDTH, HEIGHT = 604, 552
const RING_RADIUS, DISC_RADIUS = 135, 117
const STROKE = 11
const BLACK = "#000000"

# Ring and disc colours of the Julia logo.
const GREEN = ("#389826", "#60ad51")
const RED = ("#cb3c33", "#d5635c")
const PURPLE = ("#9558b2", "#aa79c1")

function draw_disc(centre, colours)
    setcolor(colours[1])
    circle(centre, RING_RADIUS, :fill)
    setcolor(colours[2])
    circle(centre, DISC_RADIUS, :fill)
end

function draw_line(points)
    setcolor(BLACK)
    poly(points, :stroke, close = false)
end

# The FFT: the butterflies of a four points Cooley-Tukey transform.
function draw_butterfly()
    columns, rows = (-74, 0, 74), (-66, -22, 22, 66)
    partners = ((2, 1, 4, 3), (3, 4, 1, 2))
    gsave()
    setline(0.6 * STROKE)
    for (stage, partner) in enumerate(partners), (row, y) in enumerate(rows)
        left, right = columns[stage], columns[stage + 1]
        draw_line([Point(left, y), Point(right, y)])
        draw_line([Point(left, y), Point(right, rows[partner[row]])])
    end
    grestore()
    setcolor(BLACK)
    for x in columns, y in rows
        circle(Point(x, y), STROKE, :fill)
    end
end

# The characteristic function of Merton with jumps of 3 and a drift of -1, in the complex plane: with next to no
# diffusion (MertonProcess wants it positive) it is periodic, with three lobes, and turned by a quarter they sit like the discs.
# The arrows are its gradient with respect to the jump intensity, what AD carries through the pricer.
function draw_characteristic_function()
    radius, intensity, arrow_length, offset = 98, 0.85, 36, 0.3
    merton(λ) = MertonProcess(1e-3, λ, 3.0, 1e-3, Underlying(100.0, 0.0))
    φ(v, λ = intensity) = exp(-im * v + characteristic_exponent_i(im * v, merton(λ), 1.0))
    ∂φ(v) = ForwardDiff.derivative(λ -> φ(v, λ), intensity)
    plane(z) = Point(-imag(z), -real(z))
    draw_line(plane.(radius .* φ.(range(0, 2π, length = 600))))
    for lobe in 0:2, side in (-1, 1)
        v = 2π * lobe / 3 + side * offset
        start = radius * φ(v)
        arrow(plane(start), plane(start + arrow_length * ∂φ(v) / abs(∂φ(v))), arrowheadlength = 14, arrowheadangle = π / 6, linewidth = 3)
    end
end

# Implied volatilities of Heston, priced by this package from out of the money options.
function heston_smile(log_strikes)
    S0, r, d, T = 100.0, 0.0, 0.0, 0.25
    model = HestonProcess(1.0, 0.2, 0.0, 1.5, -0.4, 0.04, Underlying(S0, d))
    strikes = S0 .* exp.(log_strikes)
    calls = collect(strikes .>= S0)
    prices = pricer(model, ZeroRate(r), CarrMadanMethod(400.0, 18), EuropeanOptionSmile(T, strikes, calls))
    return blsimpv.(S0, strikes, r, T, prices, d, calls)
end

# The result: the smile drawn as a spectrum, one stem for each evenly spaced log-strike, as tall as its implied volatility.
function draw_smile()
    foot, tallest = 68, 133
    xs = -72:24:72
    volatilities = heston_smile(range(-0.3, 0.3, length = length(xs)))
    tops = foot .- tallest .* volatilities ./ maximum(volatilities)
    setline(0.6 * STROKE)
    draw_line([Point(-82, foot), Point(82, foot)])
    for (x, top) in zip(xs, tops)
        draw_line([Point(x, foot), Point(x, top)])
        circle(Point(x, top), STROKE, :fill)
    end
end

function draw_symbol(draw, centre)
    gsave()
    translate(centre)
    draw()
    grestore()
end

function draw_logo()
    background(1, 1, 1, 0)
    setline(STROKE)
    setlinecap("round")
    setlinejoin("round")
    top, left, right = Point(302, 145), Point(146, 415), Point(458, 415)
    for (centre, colours, draw) in ((top, GREEN, draw_butterfly), (left, RED, draw_characteristic_function), (right, PURPLE, draw_smile))
        draw_disc(centre, colours)
        draw_symbol(draw, centre)
    end
end

# The logo as a PNG, centred on a square of the given side.
function square_png(side)
    path = tempname() * ".png"
    Drawing(side, side, path)
    setstrokescale(true)
    scale(side / WIDTH)
    translate(0, (WIDTH - HEIGHT) / 2)
    draw_logo()
    finish()
    return read(path)
end

# An ICO file is a directory of images, here PNGs, each with its size and position in the file.
function write_ico(path, sides)
    pngs = square_png.(sides)
    open(path, "w") do io
        write(io, UInt16(0), UInt16(1), UInt16(length(pngs)))
        offset = 6 + 16 * length(pngs)
        for (side, png) in zip(sides, pngs)
            write(io, UInt8(side), UInt8(side), UInt16(0), UInt16(1), UInt16(32), UInt32(length(png)), UInt32(offset))
            offset += length(png)
        end
        foreach(png -> write(io, png), pngs)
    end
end

Drawing(WIDTH, HEIGHT, joinpath(@__DIR__, "logo.png"))
draw_logo()
finish()
write_ico(joinpath(@__DIR__, "favicon.ico"), [16, 32, 48])
