function quadratic_fit_search(f, a, b, c, n)
    ya, yb, yc = f(a), f(b), f(c)
    for i in 1:n-3
        x = 0.5*(ya*(b^2-c^2)+yb*(c^2-a^2)+yc*(a^2-b^2)) / (ya*(b-c) +yb*(c-a) +yc*(a-b))
        yx = f(x)
        if x > b
            if yx > yb
                c, yc = x, yx
            else
                a, ya, b, yb = b, yb, x, yx
            end
        elseif x < b
            if yx > yb
                a, ya = x, yx
            else
                c, yc, b, yb = b, yb, x, yx
            end
        end
        print(a," ",b," ",c," ",x," ",yx,"\n")
    end
    return (a, b, c)
end

function f(n)
    return n*n+2*n-1
end

result = quadratic_fit_search(f,1,5,10,5)
















