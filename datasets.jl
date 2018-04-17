function get_random_addition()
    srand(Dates.Time(now()).nanoseconds)
    srand(rand())
    a = round(Int32, rand() * typemax(Int32)/12)
    srand(Dates.Time(now()).nanoseconds)
    srand(rand())
    b = round(Int32, rand() * typemax(Int32)/12)
    (a, b, a + b)
end


function get_addition_dataset(nr)
    empty_arr = Array{Int32, 1}(nr)
    map(x -> get_random_addition(), empty_arr)
end
