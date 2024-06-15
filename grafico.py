import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
from fractions import Fraction

class GaussianEliminationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eliminación Gaussiana con Pivoteo Parcial")
        
        # Variables para almacenar los datos de entrada
        self.n = tk.IntVar()
        self.matrix_A_entries = []
        self.vector_b_entries = []
        
        # Crear widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Etiqueta y entrada para tamaño de la matriz A
        lbl_size = tk.Label(self.root, text="Tamaño n de la matriz A (nxn):")
        lbl_size.grid(row=0, column=0, padx=10, pady=5, sticky=tk.E)
        
        entry_size = tk.Entry(self.root, textvariable=self.n, width=5)
        entry_size.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        
        # Botón para ingresar los coeficientes de la matriz A y vector b
        btn_enter = tk.Button(self.root, text="Ingresar datos", command=self.enter_data)
        btn_enter.grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
        
        # Área de texto desplazable para mostrar el procedimiento y solución
        self.txt_procedure = scrolledtext.ScrolledText(self.root, width=80, height=20, wrap=tk.WORD)
        self.txt_procedure.grid(row=1, column=0, columnspan=3, padx=10, pady=5)
        
        # Botón para resolver el sistema de ecuaciones
        btn_solve = tk.Button(self.root, text="Resolver sistema", command=self.solve_system)
        btn_solve.grid(row=2, column=1, padx=10, pady=5)
        
        # Botón para limpiar el área de texto
        btn_clear = tk.Button(self.root, text="Limpiar", command=self.clear_text)
        btn_clear.grid(row=2, column=2, padx=10, pady=5)
        
    def enter_data(self):
        try:
            n = self.n.get()
            if n <= 0:
                raise ValueError("El tamaño n debe ser mayor que cero.")
            
            # Limpiar entradas anteriores si es necesario
            self.clear_previous_entries()
            
            # Crear entradas para los coeficientes de la matriz A y vector b
            for i in range(n):
                for j in range(n):
                    lbl = tk.Label(self.root, text=f"A[{i+1},{j+1}]:")
                    lbl.grid(row=i+3, column=2*j, padx=5, pady=3, sticky=tk.E)
                    
                    entry = tk.Entry(self.root, width=5)
                    entry.grid(row=i+3, column=2*j+1, padx=5, pady=3, sticky=tk.W)
                    self.matrix_A_entries.append(entry)
            
            for i in range(n):
                lbl = tk.Label(self.root, text=f"b[{i+1}]:")
                lbl.grid(row=i+3, column=2*n, padx=5, pady=3, sticky=tk.E)
                
                entry = tk.Entry(self.root, width=5)
                entry.grid(row=i+3, column=2*n+1, padx=5, pady=3, sticky=tk.W)
                self.vector_b_entries.append(entry)
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def solve_system(self):
        try:
            n = self.n.get()
            if n <= 0:
                raise ValueError("El tamaño n debe ser mayor que cero.")
            
            # Obtener coeficientes de la matriz A
            matrix_A = []
            for i in range(n):
                row = []
                for j in range(n):
                    value = self.matrix_A_entries[i*n + j].get().strip()
                    if not value:
                        raise ValueError("Ingrese todos los coeficientes de la matriz A.")
                    row.append(int(value))
                matrix_A.append(row)
            
            # Obtener términos independientes b
            vector_b = []
            for i in range(n):
                value = self.vector_b_entries[i].get().strip()
                if not value:
                    raise ValueError("Ingrese todos los términos independientes b.")
                vector_b.append(int(value))
            
            # Resolver el sistema de ecuaciones utilizando eliminación gaussiana con pivoteo parcial
            x, procedure = self.gauss_elimination_pivot(matrix_A, vector_b)
            
            # Mostrar el procedimiento y la solución en el área de texto
            self.txt_procedure.delete('1.0', tk.END)  # Limpiar el área de texto
            self.txt_procedure.insert(tk.END, "Procedimiento de eliminación gaussiana con pivoteo parcial:\n\n")
            for step in procedure:
                self.txt_procedure.insert(tk.END, step + "\n")
            
            self.txt_procedure.insert(tk.END, "\nLa solución del sistema de ecuaciones es:\n")
            for i in range(len(x)):
                solution = Fraction(x[i]).limit_denominator()
                self.txt_procedure.insert(tk.END, f"x{i+1} = {solution}\n")
        
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def gauss_elimination_pivot(self, A, b):
        n = len(b)
        
        # Convertir la matriz A y el vector b a tipo float para evitar errores de tipo
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        # Combinar matriz A y vector b en una sola matriz aumentada Ab
        Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)
        
        # Lista para almacenar el procedimiento
        procedure = []
        
        # Función para imprimir la matriz aumentada Ab
        def print_augmented_matrix():
            procedure.append("Matriz aumentada actual:")
            for row in Ab:
                formatted_row = "  ".join(f"{Fraction(elem).limit_denominator()}" for elem in row)
                procedure.append("  " + formatted_row)
            procedure.append("")  # Separador de líneas
        
        # Inicialmente, imprimir la matriz aumentada original
        procedure.append("Matriz aumentada inicial:")
        print_augmented_matrix()
        
        # Eliminación gaussiana con pivoteo parcial
        for i in range(n):
            # Pivoteo parcial: intercambiar filas si es necesario
            max_row = np.argmax(np.abs(Ab[i:, i])) + i
            if max_row != i:
                Ab[[i, max_row]] = Ab[[max_row, i]]
                procedure.append(f"Intercambio de filas {i+1} y {max_row+1}")
                print_augmented_matrix()
            
            # Registro del paso de eliminación gaussiana actual
            procedure.append(f"Iteración {i+1}:")
            print_augmented_matrix()
            
            # Eliminación gaussiana
            for j in range(i + 1, n):
                factor = Ab[j, i] / Ab[i, i]
                Ab[j, i:] -= factor * Ab[i, i:]
                procedure.append(f"   Fila {j+1} = Fila {j+1} - ({Fraction(factor).limit_denominator()}) * Fila {i+1}")
            print_augmented_matrix()
        
        # Sustitución hacia atrás
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (Ab[i, n] - np.dot(Ab[i, i:n], x[i:n])) / Ab[i, i]
        
        # Registrar el paso final de sustitución hacia atrás
        procedure.append("Sustitución hacia atrás:")
        for i in range(n):
            solution = Fraction(x[i]).limit_denominator()
            procedure.append(f"   x{i+1} = {solution}")
        procedure.append("")  # Separador de líneas
        
        # Devolver la solución y el procedimiento registrado
        return x, procedure
    
    def clear_previous_entries(self):
        # Limpiar entradas anteriores de la matriz A y vector b
        for entry in self.matrix_A_entries:
            entry.destroy()
        for entry in self.vector_b_entries:
            entry.destroy()
        self.matrix_A_entries.clear()
        self.vector_b_entries.clear()
    
    def clear_text(self):
        self.txt_procedure.delete('1.0', tk.END)

def main():
    root = tk.Tk()
    app = GaussianEliminationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()





