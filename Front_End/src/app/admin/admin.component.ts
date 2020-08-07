import { Component, OnInit  } from '@angular/core';
import { HttpClient } from '@angular/common/http'
import { Task } from '../service/Interfaces/task'
import { ProjetService } from '../service/ImplementationService/projet.service'
import { AuthService } from 'src/app/service/auth/auth.service';

@Component({
  selector: 'app-admin',
  templateUrl: './admin.component.html',
  styleUrls: ['./admin.component.css']
})

export class AdminComponent implements OnInit {
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
    
  }
  
  getLogin() {
    return this.authService.getUser().login;
  }



}
