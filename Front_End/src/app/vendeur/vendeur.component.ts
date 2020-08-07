import { Component, OnInit  } from '@angular/core';
import { HttpClient } from '@angular/common/http'
import { Task } from '../service/Interfaces/task'
import { ProjetService } from '../service/ImplementationService/projet.service'
import { AuthService } from 'src/app/service/auth/auth.service';

@Component({
  selector: 'app-vendeur',
  templateUrl: './vendeur.component.html',
  styleUrls: ['./vendeur.component.css'],
  providers: [ProjetService]
})

export class VendeurComponent implements OnInit {
  tasks: Task[]
  editTask: Task
  p: number=1;
  search: string;
  public _opened: boolean = false;
  public _toggleSidebar() {
    this._opened = !this._opened;
  }
  
  constructor(private projetService: ProjetService, private http: HttpClient,private authService: AuthService) { }

  ngOnInit() {
    /*this.getTasks()*/
  }
  getLogin() {
    return this.authService.getUser().login;
  }

  logout() {
    return this.authService.logout();
  }
  /*getTasks (): void {
    this.projetService.getTasks().subscribe(tasks => (this.tasks = tasks))
  }
  
  add (Product_name: string): void {
    this.editTask = undefined
    Product_name = Product_name.trim()
    if (!Product_name) {
        return
    }
    const newTask: Task = { Product_name } as Task
    this.projetService.addTask(newTask).subscribe(() => this.getTasks())
  }

  delete (task: Task): void {
    this.tasks = this.tasks.filter(h => h !== task)
    this.projetService
        .deleteTask(task._id)
        .subscribe(() => console.log('Task Deleted'))
  }
  edit (task) {
    this.editTask = task
  }

  update () {
    if (this.editTask) {
        this.projetService.updateTask(this.editTask).subscribe(() => {
            this.getTasks()
        })
        this.editTask = undefined
    }
  }*/





}
