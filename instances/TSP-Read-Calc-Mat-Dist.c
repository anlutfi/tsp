//
//  Matriz de Distancias -- TSP
//
//
//  O cálculo da distâncias entre 2 cidades deve ser feito
//  da seguinte forma:
//     1 - Calcular a distancia euclidiana em valor real (float ou double)
//     2 - Somar 0.5 a essa distancia
//     3 - Truncar para inteiro
//
//  O programa abaixo le um arquivo .tsp em coordenadas e
//  calcula a matriz de distancias.
//  Não é obrigatorio usar o codigo abaixo, desde que o
//  calculo acima.
//


int Ncities;
int Cost[MaxCities][MaxCities];

int readTSPcoord()
{
  int i, j, b;
  int dx, dy;
  double xd, yd, zd;
  FILE *p_in;

  p_in = fopen("data.tsp","r");

  fscanf(p_in, "%d", &b);
  Ncities = b + 1;

  for (i=1; i<=b; i++)
    {
      fscanf(p_in,"%d %d %d", &j, &dx, &dy);
      CoordX[i]=(float)dx;
      CoordY[i]=(float)dy;
    }
  CoordX[Ncities]=CoordX[1];
  CoordY[Ncities]=CoordY[1];


  for (i=1; i<=Ncities; i++)
    {
      for (j=1; j<=Ncities; j++)
	  {
          xd = CoordX[i] - CoordX[j];
          yd = CoordY[i] - CoordY[j];
          zd = sqrt(xd*xd+yd*yd) + 0.5;
          Cost[i][j] = (int)floor(zd);
	  }
    }

  fclose(p_in);

  return (0);

} /* end read */

