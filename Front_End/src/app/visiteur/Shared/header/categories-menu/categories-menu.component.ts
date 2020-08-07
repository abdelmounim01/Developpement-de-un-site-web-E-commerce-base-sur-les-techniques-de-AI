import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ProjetService } from '../../../../service/ImplementationService/projet.service'
import { Task } from 'src/app/service/Interfaces/task';

@Component({
  selector: 'app-categories-menu-visiteur',
  templateUrl: './categories-menu.component.html',
  styleUrls: ['./categories-menu.component.css']
})
export class CategoriesMenuvisiteurComponent implements OnInit {
  tasks: Task[]; 

  constructor(private resourceService:ProjetService , public router:Router) { }

  ngOnInit() {
    this.getInfoCategie() ; 
  }
  getInfoCategie():void{
    this.resourceService.getInfoCategie().subscribe(tasks => (this.tasks = tasks))

  }
}
