package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type Cube struct {
	Pieces   [3][3][3]Piece
	IsSolved bool
}
type Piece struct {
	Id       int
	ColorMap map[rune]int // Each color has a face assigned to it
}
type Move struct {
	Axis      int
	Line      int
	Direction bool
}
type Embed struct {
	Locations [20][3]int
	Rotations [20][6]int
}
type EmbedAbs struct {
	Embed [36]int
}
type EmbedMoves struct {
	Embed  EmbedAbs
	NMoves int
}

var initialCube = [][][][]rune{
	{
		{{'w', 'b', 'r'}, {'w', 'r'}, {'w', 'g', 'r'}},
		{{'w', 'b'}, {'w'}, {'w', 'g'}},
		{{'w', 'b', 'o'}, {'w', 'o'}, {'w', 'g', 'o'}},
	}, {
		{{'b', 'r'}, {'r'}, {'g', 'r'}},
		{{'b'}, {}, {'g'}},
		{{'b', 'o'}, {'o'}, {'g', 'o'}},
	}, {
		{{'y', 'b', 'r'}, {'y', 'r'}, {'y', 'g', 'r'}},
		{{'y', 'b'}, {'y'}, {'y', 'g'}},
		{{'y', 'b', 'o'}, {'y', 'o'}, {'y', 'g', 'o'}},
	},
}
var initialFaceColors = map[rune]int{
	'w': 0,
	'y': 1,
	'b': 2,
	'g': 3,
	'r': 4,
	'o': 5,
}
var pieceToFace = map[[3]int]map[int][2]int{
	{0, 0, 0}: {0: {0, 0}, 2: {0, 2}, 4: {2, 0}},
	{0, 0, 1}: {0: {0, 1}, 4: {2, 1}},
	{0, 0, 2}: {0: {0, 2}, 3: {0, 0}, 4: {2, 2}},
	{0, 1, 0}: {0: {1, 0}, 2: {1, 2}},
	{0, 1, 1}: {0: {1, 1}},
	{0, 1, 2}: {0: {1, 2}, 3: {1, 0}},
	{0, 2, 0}: {0: {2, 0}, 2: {2, 2}, 5: {0, 0}},
	{0, 2, 1}: {0: {2, 1}, 5: {0, 1}},
	{0, 2, 2}: {0: {2, 2}, 3: {2, 0}, 5: {0, 2}},
	{1, 0, 0}: {2: {0, 1}, 4: {1, 0}},
	{1, 0, 1}: {4: {1, 1}},
	{1, 0, 2}: {3: {0, 1}, 4: {1, 2}},
	{1, 1, 0}: {2: {1, 1}},
	{1, 1, 1}: {},
	{1, 1, 2}: {3: {1, 1}},
	{1, 2, 0}: {2: {2, 1}, 5: {1, 0}},
	{1, 2, 1}: {5: {1, 1}},
	{1, 2, 2}: {3: {2, 1}, 5: {1, 2}},
	{2, 0, 0}: {2: {0, 0}, 4: {0, 0}, 1: {0, 2}},
	{2, 0, 1}: {4: {0, 1}, 1: {0, 1}},
	{2, 0, 2}: {1: {0, 0}, 3: {0, 2}, 4: {0, 2}},
	{2, 1, 0}: {2: {1, 0}, 1: {1, 2}},
	{2, 1, 1}: {1: {1, 1}},
	{2, 1, 2}: {1: {1, 0}, 3: {1, 2}},
	{2, 2, 0}: {2: {2, 0}, 1: {2, 2}, 5: {2, 0}},
	{2, 2, 1}: {5: {2, 1}, 1: {2, 1}},
	{2, 2, 2}: {1: {2, 0}, 3: {2, 2}, 5: {2, 2}},
}
var zFaces = map[bool][6]int{
	true:  {0, 1, 4, 5, 3, 2},
	false: {0, 1, 5, 4, 2, 3},
}
var yFaces = map[bool][6]int{
	true:  {4, 5, 2, 3, 1, 0},
	false: {5, 4, 2, 3, 0, 1},
}
var xFaces = map[bool][6]int{
	true:  {3, 2, 0, 1, 4, 5},
	false: {2, 3, 1, 0, 4, 5},
}
var blank string = "         "
var MoveNotation = map[Move]string{
	{Axis: 0, Line: 0, Direction: false}: "U",
	{Axis: 0, Line: 0, Direction: true}:  "U'",
	{Axis: 0, Line: 1, Direction: false}: "E'",
	{Axis: 0, Line: 1, Direction: true}:  "E",
	{Axis: 0, Line: 2, Direction: false}: "D'",
	{Axis: 0, Line: 2, Direction: true}:  "D",
	{Axis: 1, Line: 0, Direction: false}: "L",
	{Axis: 1, Line: 0, Direction: true}:  "L'",
	{Axis: 1, Line: 1, Direction: false}: "M",
	{Axis: 1, Line: 1, Direction: true}:  "M'",
	{Axis: 1, Line: 2, Direction: false}: "R'",
	{Axis: 1, Line: 2, Direction: true}:  "R",
	{Axis: 2, Line: 0, Direction: false}: "F'",
	{Axis: 2, Line: 0, Direction: true}:  "F",
	{Axis: 2, Line: 1, Direction: false}: "S'",
	{Axis: 2, Line: 1, Direction: true}:  "S",
	{Axis: 2, Line: 2, Direction: false}: "B",
	{Axis: 2, Line: 2, Direction: true}:  "B'",
}
var centerPieces = [6][3]int{
	{0, 1, 1}, {2, 1, 1}, {1, 1, 0},
	{1, 1, 2}, {1, 0, 1}, {1, 2, 1},
}
var flatLocations = map[[3]int]int{
	{0, 0, 0}: 0,
	{0, 0, 1}: 1,
	{0, 0, 2}: 2,
	{0, 1, 0}: 3,
	{0, 1, 2}: 4,
	{0, 2, 0}: 5,
	{0, 2, 1}: 6,
	{0, 2, 2}: 7,
	{1, 0, 0}: 8,
	{1, 0, 2}: 9,
	{1, 2, 0}: 10,
	{1, 2, 2}: 11,
	{2, 0, 0}: 12,
	{2, 0, 1}: 13,
	{2, 0, 2}: 14,
	{2, 1, 0}: 15,
	{2, 1, 2}: 16,
	{2, 2, 0}: 17,
	{2, 2, 1}: 18,
	{2, 2, 2}: 19,
}
var faceDistances = map[[2]int]int{
	{0, 0}: 0, {1, 0}: 2, {2, 0}: 1, {3, 0}: -1, {4, 0}: -1, {5, 0}: 1,
	{0, 1}: -2, {1, 1}: 0, {2, 1}: -1, {3, 1}: 1, {4, 1}: 1, {5, 1}: -1,
	{0, 2}: -1, {1, 2}: 1, {2, 2}: 0, {3, 2}: 2, {4, 2}: -1, {5, 2}: 1,
	{0, 3}: 1, {1, 3}: -1, {2, 3}: -2, {3, 3}: 0, {4, 3}: 1, {5, 3}: -1,
	{0, 4}: 1, {1, 4}: -1, {2, 4}: 1, {3, 4}: -1, {4, 4}: 0, {5, 4}: 2,
	{0, 5}: -1, {1, 5}: 1, {2, 5}: -1, {3, 5}: 1, {4, 5}: -2, {5, 5}: 0,
}
var faceAxis = map[int]int{
	0: 2,
	1: 2,
	2: 1,
	3: 1,
	4: 0,
	5: 0,
}

func getInitialRotations(colors []rune) map[rune]int {
	colorMap := make(map[rune]int)

	for _, color := range colors {
		colorMap[color] = initialFaceColors[color]
	}

	return colorMap
}

func initializeCube() Cube {
	// The array is face, line, column
	// The faces are ordered front, back, left, right, top, bottom
	var cube Cube

	count := 0
	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				colors := initialCube[layer][row][col]
				colorMap := getInitialRotations(colors)
				cube.Pieces[layer][row][col] = Piece{ColorMap: colorMap, Id: count}
				count++
			}
		}
	}
	cube.IsSolved = true

	return cube
}

func moveZ(cube Cube, section int, direction bool) Cube {
	var newPiece Piece
	_cube := cube

	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			// Select the new piece for the location
			if direction {
				newPiece = cube.Pieces[section][2-col][row]
			} else {
				newPiece = cube.Pieces[section][col][2-row]
			}

			// Rotate new piece accordingly
			for color, r := range newPiece.ColorMap {
				newPiece.ColorMap[color] = zFaces[direction][r]
			}

			// Substitute the piece
			_cube.Pieces[section][row][col] = newPiece
		}
	}

	return _cube
}

func moveY(cube Cube, col int, direction bool) Cube {
	var newPiece Piece
	_cube := cube

	for section := 0; section < 3; section++ {
		for row := 0; row < 3; row++ {
			// Select the new piece for the location
			if direction {
				newPiece = cube.Pieces[row][2-section][col]
			} else {
				newPiece = cube.Pieces[2-row][section][col]
			}

			// Rotate new piece accordingly
			for color, r := range newPiece.ColorMap {
				newPiece.ColorMap[color] = yFaces[direction][r]
			}

			// Substitute the piece
			_cube.Pieces[section][row][col] = newPiece
			// cube, moves := scrambleCube(cube, 5)
		}
	}

	return _cube
}

func moveX(cube Cube, row int, direction bool) Cube {
	var newPiece Piece
	_cube := cube

	for section := 0; section < 3; section++ {
		for col := 0; col < 3; col++ {
			// Select the new piece for the location
			if direction {
				newPiece = cube.Pieces[2-col][row][section]
			} else {
				newPiece = cube.Pieces[col][row][2-section]
			}

			// Rotate new piece accordingly
			for color, r := range newPiece.ColorMap {
				newPiece.ColorMap[color] = xFaces[direction][r]
			}

			// Substitute the piece
			_cube.Pieces[section][row][col] = newPiece
		}
	}

	return _cube
}

func MoveCube(cube Cube, move Move) Cube {
	_cube := cube

	switch move.Axis {
	case 0:
		_cube = moveX(_cube, move.Line, move.Direction)
	case 1:
		_cube = moveY(_cube, move.Line, move.Direction)
	case 2:
		_cube = moveZ(_cube, move.Line, move.Direction)
	}

	_cube = CheckSolvedCube(_cube)
	return _cube
}

func RotateCube(cube Cube, axis int, direction bool) Cube {
	_cube := cube

	for line := 0; line < 3; line++ {
		move := Move{Axis: axis, Line: line, Direction: direction}
		_cube = MoveCube(_cube, move)
	}

	return _cube
}

func GetFaceColors(cube Cube) map[rune]int {
	faceColors := make(map[rune]int)

	for _, loc := range centerPieces {
		piece := cube.Pieces[loc[0]][loc[1]][loc[2]]
		for color, r := range piece.ColorMap {
			faceColors[color] = r
		}
	}

	return faceColors
}

func GetCorrectLocation(piece Piece, faceColors map[rune]int) [3]int {
	var face int
	location := [3]int{1, 1, 1}

	for color := range piece.ColorMap {
		face = faceColors[color]

		switch face {
		case 0:
			location[0] = 0
		case 1:
			location[0] = 2
		case 2:
			location[2] = 0
		case 3:
			location[2] = 2
		case 4:
			location[1] = 0
		case 5:
			location[1] = 2
		}
	}

	return location
}

func CheckCorrectLocation(piece Piece, faceColors map[rune]int) bool {
	for color, r := range piece.ColorMap {
		if faceColors[color] != r {
			return false
		}
	}

	return true
}

func CheckSolvedCube(cube Cube) Cube {
	_cube := cube
	faceColors := GetFaceColors(_cube)

	for section := 0; section < 3; section++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				piece := _cube.Pieces[section][row][col]
				if !CheckCorrectLocation(piece, faceColors) {
					_cube.IsSolved = false
					return _cube
				}
			}
		}
	}

	_cube.IsSolved = true
	return _cube
}

func scrambleCube(cube Cube, nMoves int) (Cube, []string) {
	var moves []string
	_cube := cube

	for i := 0; i < nMoves; i++ {
		move := Move{rand.Intn(3), rand.Intn(3), rand.Float32() <= 0.5}
		moves = append(moves, MoveNotation[move])
		_cube = MoveCube(_cube, move)
	}
	return _cube, moves
}

func InitializeScrambledCube(nMoves int) Cube {
	cube := initializeCube()

	cube, _ = scrambleCube(cube, nMoves)

	return cube
}

func stringifyLine(line [3]rune) string {
	var strLine string
	for i := 0; i < 3; i++ {
		strLine += "|" + string(line[i]) + "|"
	}
	return strLine
}

func CubeToFaces(cube Cube) [6][3][3]rune {
	// 		|---|
	// 		|-4-|
	//      |---|
	// |---||---||---||---|
	// |-2-||-0-||-3-||-1-|
	// |---||---||---||---|
	// 		|---|
	// 		|-5-|
	// 		|---|
	var faces [6][3][3]rune

	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				piece := cube.Pieces[layer][row][col]
				faceMap := pieceToFace[[3]int{layer, row, col}]
				for color, r := range piece.ColorMap {
					coord := faceMap[r]
					faces[r][coord[0]][coord[1]] = color
				}
			}
		}
	}

	return faces
}

func PrintCube(cube Cube) {

	faces := CubeToFaces(cube)

	// top face
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(faces[4][i]))
	}
	fmt.Println()

	// left, front, right and back face
	for i := 0; i < 3; i++ {
		fmt.Println(
			stringifyLine(faces[2][i]),
			stringifyLine(faces[0][i]),
			stringifyLine(faces[3][i]),
			stringifyLine(faces[1][i]),
		)
	}
	fmt.Println()

	// top bottomPosition
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(faces[5][i]))
	}
}

func EmbedCube(cube Cube) Embed {

	var embed Embed
	embed.Rotations = [20][6]int{}

	faceColors := GetFaceColors(cube)

	for key, val := range flatLocations {
		// Piece & Locations
		piece := cube.Pieces[key[0]][key[1]][key[2]]
		correctLocation := GetCorrectLocation(piece, faceColors)

		// Calculate location distances
		for i, loc := range correctLocation {
			embed.Locations[val][i] = loc - key[i]
		}

		// Calculate rotation distances
		for color, r := range piece.ColorMap {
			embed.Rotations[val][initialFaceColors[color]] = faceDistances[[2]int{faceColors[color], r}]
		}
	}
	return embed
}

func GetFaceAbsDist(cube Cube) [7]int {
	var faceAbsDist [7]int
	var locationDist int

	faceColors := GetFaceColors(cube)

	for key, _ := range flatLocations {
		// Piece & Locations
		piece := cube.Pieces[key[0]][key[1]][key[2]]
		correctLocation := GetCorrectLocation(piece, faceColors)

		// Calculate location distances
		for i, loc := range correctLocation {
			locationDist += int(math.Abs(float64(loc - key[i])))
		}

		// Calculate rotation distances
		for color, r := range piece.ColorMap {
			faceAbsDist[faceColors[color]] += int(math.Abs(float64(faceDistances[[2]int{faceColors[color], r}]))) + locationDist
		}
	}

	faceAbsDist[6] = math.MaxInt

	return faceAbsDist
}

func GetSmallestFaceDist(cube Cube) [3]int {
	faceAbsDist := GetFaceAbsDist(cube)

	smallest := [3]int{6, 6, 6}
	for i, val := range faceAbsDist {
		if val <= faceAbsDist[smallest[0]] {
			smallest[2] = smallest[1]
			smallest[1] = smallest[0]
			smallest[0] = i
		} else if val <= faceAbsDist[smallest[1]] {
			smallest[2] = smallest[1]
			smallest[1] = i
		} else if val <= faceAbsDist[smallest[2]] {

		}
	}

	return smallest
}

// func StandardRotate(cube Cube) Cube {
// 	var Axis [3]int
// 	_cube := cube

// 	// Determine the two contiguous smallest faces
// 	smallest := GetSmallestFaceDist(_cube)
// 	frontFace := smallest[0]

// 	topFace := smallest[1]
// 	if math.Floor(float64(frontFace)/2) == math.Floor(float64(topFace)/2) {
// 		topFace = smallest[2]
// 	}

// 	// Get the face axis for each of the smallest faces
// 	Axis[0] = faceAxis[frontFace]
// 	Axis[1] = faceAxis[topFace]
// 	for i := 0; i < 3; i++ {
// 		if (i != Axis[0]) && (i != Axis[1]) {
// 			Axis[2] = i
// 			break
// 		}
// 	}

// 	// Move to the standard position
// 	if Axis[2] == 2 {

// 	} else {

// 	}

// }

func AbsoluteEmbed(cube Cube) EmbedAbs {
	var prev mat.Dense
	var embedMat mat.Dense
	var embed EmbedAbs
	var locations [20 * 3]float64
	rotations := [20 * 6]float64{}

	faceColors := GetFaceColors(cube)

	for key, val := range flatLocations {
		// Piece & Locations
		piece := cube.Pieces[key[0]][key[1]][key[2]]
		correctLocation := GetCorrectLocation(piece, faceColors)

		// Calculate location distances
		for i, loc := range correctLocation {
			locations[val*3+i] = math.Abs(float64(loc - key[i]))
		}

		// Calculate rotation distances
		for color, r := range piece.ColorMap {
			rotations[val*6+initialFaceColors[color]] = math.Abs(float64(faceDistances[[2]int{faceColors[color], r}]))
		}
	}

	locMat := mat.NewDense(20, 3, locations[:])
	rotMat := mat.NewDense(20, 6, rotations[:])

	prev.Mul(locMat.T(), rotMat)
	embedMat.Mul(prev.T(), &prev)

	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			embed.Embed[i*6+j] = int(embedMat.At(i, j))
		}
	}

	return embed
}

func scrambleEmbed(max_moves int, randomize_nMoves bool, c chan EmbedMoves) {
	nMoves := max_moves
	if randomize_nMoves {
		if max_moves < 2 {
			nMoves = 1
		} else {
			nMoves = rand.Intn(max_moves-1) + 1
		}
	}

	cube := InitializeScrambledCube(nMoves)
	embedMoves := EmbedMoves{Embed: AbsoluteEmbed(cube), NMoves: nMoves}

	c <- embedMoves
}

func generateEmbeds(steps int, max_moves int, randomize_nMoves bool, c chan EmbedMoves) {

	for i := 0; i < steps; i++ {
		go scrambleEmbed(max_moves, randomize_nMoves, c)
	}
}

func collectEmbeds(fileName string, steps int, step_size int, lower int, upper int) {
	var embed EmbedMoves
	var newMoves int

	c := make(chan EmbedMoves, 100)

	bestSolves := make(map[EmbedAbs]int)
	for max_moves := upper; max_moves >= lower; max_moves-- {
		// Generate new solves
		for step := 0; step < steps; step++ {
			fmt.Printf("\n---- Starting step %v of %v ----\n", step, steps)

			// prev_solves := len(bestSolves)
			go generateEmbeds(step_size, max_moves, false, c)
			for i := 0; i < step_size; i++ {
				embed = <-c
				newMoves = embed.NMoves
				bestSolves[embed.Embed] = newMoves

				if i%1000 == 0 {
					percent := float32(i) / float32(step_size) * 100
					fmt.Printf("Max_Moves: %v \t\t Step: %v \t\t -- %.2f%% --\t\t Samples: %v\n", max_moves, i, percent, len(bestSolves))
				}
			}

			// if prev_solves == len(bestSolves) {
			// 	break
			// }

		}

	}

	// Write
	fmt.Print("\nWriting new solves...\n\n")
	writeAbsSolves(bestSolves, fileName)
}

func createWriter(file string) *bufio.Writer {
	osFile, errWrite := os.OpenFile(file, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if errWrite != nil {
		log.Fatal(errWrite)
	}

	Writer := bufio.NewWriter(osFile)

	return Writer
}

func writeSolves(solves map[EmbedAbs]int, fileName string) {
	file := make(map[string]string)
	file["features"] = fileName + "features.json"
	file["labels"] = fileName + "labels.csv"

	os.Remove(file["features"])
	os.Remove(file["labels"])

	featWriter := createWriter(file["features"])
	labWriter := createWriter(file["labels"])
	defer featWriter.Flush()
	defer labWriter.Flush()

	featWriter.WriteString("[")
	first := true
	for embed, nMoves := range solves {

		// Write the number of moves
		labWriter.WriteString(strconv.Itoa(nMoves) + "\n")

		// Write the embeddings
		b, err := json.Marshal(embed)
		if err != nil {
			fmt.Println(err)
			return
		}

		if first {
			featWriter.WriteString(string(b))
			first = false
		} else {
			featWriter.WriteString(",\n" + string(b))
		}

		delete(solves, embed)
	}
	featWriter.WriteString("]")
}

func arrayToString(a []int, delim string) string {
	return strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]")
	//return strings.Trim(strings.Join(strings.Split(fmt.Sprint(a), " "), delim), "[]")
	//return strings.Trim(strings.Join(strings.Fields(fmt.Sprint(a)), delim), "[]")
}

func writeAbsSolves(solves map[EmbedAbs]int, fileName string) {
	file := make(map[string]string)
	file["features"] = fileName + "features.csv"
	file["labels"] = fileName + "labels.csv"

	os.Remove(file["features"])
	os.Remove(file["labels"])

	featWriter := createWriter(file["features"])
	labWriter := createWriter(file["labels"])
	defer featWriter.Flush()
	defer labWriter.Flush()

	for embed, nMoves := range solves {

		// Write the number of moves
		labWriter.WriteString(strconv.Itoa(nMoves) + "\n")

		// Write the embeddings

		featWriter.WriteString(arrayToString(embed.Embed[:], ",") + "\n")

		delete(solves, embed)
	}
}

// func readSolves(file string, maxMoves int) map[Embed]int {
// 	var embed Embed
// 	var nMoves int
// 	bestSolves := make(map[Embed]int)
// 	// n_embed := len(embed.Embed)

// 	// Check for saved solves
// 	csvReader, fRead := createReader(file)
// 	for {
// 		rec, err := csvReader.Read()
// 		if err == io.EOF {
// 			break
// 		}
// 		if err != nil {
// 			log.Fatal(err)
// 		}

// 		for i, s := range rec {
// 			if i < n_embed {
// 				embed.Embed[i], _ = strconv.Atoi(s)
// 			} else {
// 				nMoves, _ = strconv.Atoi(s)
// 			}
// 		}
// 		if nMoves <= maxMoves {
// 			bestSolves[embed] = nMoves
// 		}
// 	}
// 	fRead.Close()

// 	return bestSolves
// }

func createReader(file string) (*csv.Reader, *os.File) {
	fRead, errRead := os.Open(file)
	if errRead != nil {
		log.Fatal(errRead)
	}

	return csv.NewReader(fRead), fRead
}

func main() {
	// var embed Embed
	// bestSolves := make(map[Embed]int)

	sourceFile := "solves/solves_"
	// destinationFile := "solves/solves_"

	// Create new solves (Generate maximum data for few moves)
	total_iter := 1e6
	step_size := 10000
	steps := int(float64(total_iter) / float64(step_size))

	collectEmbeds(sourceFile, steps, step_size, 1, 25)

	// // Prune solves
	// os.Remove(destinationFile + "features.csv")
	// os.Remove(destinationFile + "labels.csv")

	// rawFeatReader, fFile := createReader(sourceFile + "features.csv")
	// rawLabReader, lFile := createReader(sourceFile + "labels.csv")

	// step := 0
	// for {
	// 	// Read feature row
	// 	feature, err := rawFeatReader.Read()
	// 	if err == io.EOF {
	// 		break
	// 	}
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}

	// 	// Read label row
	// 	label, err := rawLabReader.Read()
	// 	if err == io.EOF {
	// 		break
	// 	}
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}

	// 	for i, s := range feature {
	// 		embed.Embed[i], _ = strconv.Atoi(s)
	// 	}
	// 	newMoves, _ := strconv.Atoi(label[0])

	// 	if oldMoves, ok := bestSolves[embed]; ok {
	// 		if newMoves < oldMoves {
	// 			bestSolves[embed] = newMoves
	// 		}
	// 	} else {
	// 		bestSolves[embed] = newMoves
	// 	}

	// 	if step%100000 == 0 {
	// 		fmt.Printf("Processed %v rows\n", step)
	// 	}
	// 	step++
	// }

	// fFile.Close()
	// lFile.Close()

	// fmt.Println("Starting writting...")
	// writeSolves(bestSolves, destinationFile)
}
