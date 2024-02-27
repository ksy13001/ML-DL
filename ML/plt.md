1. 방정식 그래프
   plt.plot(x, y)

   ex) y  = 3*x + 5,  -100 <= x <= 100
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   x = np.array(range(-100, 100))
   plt.plot(x, 3*x+5)
   plt.show()
   ```

   ex) y = x^3 + 5 * x^2 + 7,  -100 <= x <= 100
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   x = np.array(range(-100, 100))
   plt.plot(x, x**3 + 5*x**2 + 7)
   plt.show()
   ```
