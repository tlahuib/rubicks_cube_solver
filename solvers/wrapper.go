package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rubik"
)

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}

func encodeCube(cube rubik.Cube) {

	b, err := json.Marshal(cube)
	handleErr(err)

	fmt.Print(b)
	fmt.Print("|")
	fmt.Print(rubik.EmbedCube(cube))
	fmt.Print("\n")

}

func encodeCubeDiff(cube rubik.Cube, newCube rubik.Cube) {
	b, err := json.Marshal(newCube)
	handleErr(err)
	embed := rubik.EmbedCube(cube)
	newEmbed := rubik.EmbedCube(newCube)
	diff := rubik.CompareEmbeddings(embed, newEmbed)

	fmt.Print(b)
	fmt.Print("|")
	fmt.Print(newEmbed)
	fmt.Print("|")
	fmt.Print(append(newEmbed, diff...))
	fmt.Print("\n")

}

func decodeCube(eCube string) rubik.Cube {
	var cube rubik.Cube
	var bCube []byte

	sCubeBytes := strings.Split(eCube, " ")
	for _, sByte := range sCubeBytes {
		iByte, err := strconv.Atoi(sByte)
		handleErr(err)
		bCube = append(bCube, byte(iByte))
	}

	json.Unmarshal(bCube, &cube)

	return cube
}

func receiveRandomCube(sMoves string) {
	nMoves, err := strconv.Atoi(sMoves)
	handleErr(err)

	cube := rubik.InitializeScrambledCube(nMoves)

	encodeCube(cube)
}

func getPossibleMoves(eCube string) {

	cube := decodeCube(eCube)

	for _, newCube := range rubik.GetPossibleMoves(cube) {
		encodeCubeDiff(cube, newCube)
	}

}

func main() {
	args := os.Args[1:]

	switch f := args[0]; f {
	case "receiveRandomCube":
		receiveRandomCube(args[1])
	case "getPossibleMoves":
		getPossibleMoves(args[1])
	default:
		fmt.Println("No valid function selected.")
	}
}
