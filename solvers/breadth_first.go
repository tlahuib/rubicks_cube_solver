package main

import (
	"fmt"
	"time"

	cc "github.com/cube_constructor"
)

func visitNextStates(state cc.Cube, history []string) map[cc.Cube][]string {
	proposals := make(map[cc.Cube][]string)

	for axis := 0; axis < 3; axis++ {
		for line := 0; line < 3; line++ {
			for dir := 0; dir < 2; dir++ {
				move := cc.Move{Axis: axis, Line: line, Direction: dir == 0}
				proposals[cc.MoveCube(state, move)] = append(history, cc.MoveNotation[move])
			}
		}
	}

	return proposals
}

func PrintStates(states *int, cycle *int) {
	fmt.Printf("Number of Cycles: %v\nNumber of States: %v\n", *cycle, *states)
}

func pingStatus(states *int, cycle *int) {
	for {
		fmt.Println("Performing Status check:")
		PrintStates(states, cycle)
		time.Sleep(time.Second * 1)
	}
}

func main() {
	var solution cc.Cube
	states := make(map[cc.Cube][]string)
	n := 10
	cube := cc.InitializeScrambledCube(n)
	queue := []cc.Cube{cube}

	states[cube] = []string{}

	i := 0
	statesVisited := 0
	go pingStatus(&statesVisited, &i)

solver:
	for ; i < n; i++ {
		for _, sourceCube := range queue {
			prospects := visitNextStates(sourceCube, states[sourceCube])
			for prospect, history := range prospects {
				// Update if state is new
				if states[prospect] == nil {
					// Update map
					states[prospect] = history

					// Check if solved
					if prospect.IsSolved {
						solution = prospect
						break solver
					}

					// Update Queue
					statesVisited += 1
					queue = append(queue, prospect)
				}
			}
			// Remove from Queue
			queue = queue[1:]
		}
	}

	fmt.Println(states[solution])
	cc.PrintCube(solution)
}
